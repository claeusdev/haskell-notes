-- ExpressionEvaluator.hs
-- A comprehensive expression evaluator demonstrating pattern matching and recursion
-- Features: parsing, evaluation, optimization, variables, and functions

{-# LANGUAGE LambdaCase #-}

module ExpressionEvaluator where

import Data.Char (isSpace, isAlpha, isDigit, isAlphaNum)
import Data.List (intercalate)
import qualified Data.Map as Map
import Control.Monad (when)

-- Data types representing expressions
data Expr 
    = Num Double
    | Var String
    | Add Expr Expr
    | Sub Expr Expr
    | Mul Expr Expr
    | Div Expr Expr
    | Pow Expr Expr
    | Neg Expr
    | Call String [Expr]  -- Function calls
    deriving (Show, Eq)

-- Environment for variables and functions
type VarEnv = Map.Map String Double
type FuncEnv = Map.Map String ([String], Expr)  -- (parameters, body)

data EvalEnv = EvalEnv
    { variables :: VarEnv
    , functions :: FuncEnv
    } deriving (Show)

-- Error types for better error handling
data EvalError
    = DivisionByZero
    | UnknownVariable String
    | UnknownFunction String
    | TypeError String
    | ParseError String
    | ArityError String Int Int  -- function name, expected, actual
    deriving (Show, Eq)

type EvalResult = Either EvalError Double

-- Parser data types
data Token 
    = TNum Double
    | TVar String
    | TPlus | TMinus | TMul | TDiv | TPow
    | TLParen | TRParen
    | TComma | TEquals
    | TEOF
    deriving (Show, Eq)

-- Tokenization
tokenize :: String -> [Token]
tokenize [] = [TEOF]
tokenize (c:cs)
    | isSpace c = tokenize cs
    | c == '+' = TPlus : tokenize cs
    | c == '-' = TMinus : tokenize cs
    | c == '*' = TMul : tokenize cs
    | c == '/' = TDiv : tokenize cs
    | c == '^' = TPow : tokenize cs
    | c == '(' = TLParen : tokenize cs
    | c == ')' = TRParen : tokenize cs
    | c == ',' = TComma : tokenize cs
    | c == '=' = TEquals : tokenize cs
    | isDigit c = parseNumber (c:cs)
    | isAlpha c = parseIdentifier (c:cs)
    | otherwise = error $ "Unexpected character: " ++ [c]
  where
    parseNumber str = 
        let (numStr, rest) = span (\x -> isDigit x || x == '.') str
        in TNum (read numStr) : tokenize rest
    
    parseIdentifier str =
        let (identStr, rest) = span isAlphaNum str
        in TVar identStr : tokenize rest

-- Recursive descent parser
type Parser = [Token] -> Either EvalError (Expr, [Token])

-- Parse expression (handles addition and subtraction)
parseExpr :: Parser
parseExpr tokens = do
    (left, rest1) <- parseTerm tokens
    parseExprRest left rest1
  where
    parseExprRest left (TPlus:rest) = do
        (right, rest2) <- parseTerm rest
        parseExprRest (Add left right) rest2
    parseExprRest left (TMinus:rest) = do
        (right, rest2) <- parseTerm rest
        parseExprRest (Sub left right) rest2
    parseExprRest left rest = Right (left, rest)

-- Parse term (handles multiplication and division)
parseTerm :: Parser
parseTerm tokens = do
    (left, rest1) <- parseFactor tokens
    parseTermRest left rest1
  where
    parseTermRest left (TMul:rest) = do
        (right, rest2) <- parseFactor rest
        parseTermRest (Mul left right) rest2
    parseTermRest left (TDiv:rest) = do
        (right, rest2) <- parseFactor rest
        parseTermRest (Div left right) rest2
    parseTermRest left rest = Right (left, rest)

-- Parse factor (handles exponentiation, unary minus, parentheses)
parseFactor :: Parser
parseFactor (TMinus:rest) = do
    (expr, rest2) <- parseFactor rest
    Right (Neg expr, rest2)
parseFactor tokens = do
    (left, rest1) <- parseAtom tokens
    parseFactorRest left rest1
  where
    parseFactorRest left (TPow:rest) = do
        (right, rest2) <- parseFactor rest  -- Right associative
        Right (Pow left right, rest2)
    parseFactorRest left rest = Right (left, rest)

-- Parse atomic expressions (numbers, variables, function calls, parentheses)
parseAtom :: Parser
parseAtom (TNum n:rest) = Right (Num n, rest)
parseAtom (TVar name:TLParen:rest) = parseFunctionCall name rest
parseAtom (TVar name:rest) = Right (Var name, rest)
parseAtom (TLParen:rest) = do
    (expr, rest2) <- parseExpr rest
    case rest2 of
        TRParen:rest3 -> Right (expr, rest3)
        _ -> Left $ ParseError "Expected closing parenthesis"
parseAtom [] = Left $ ParseError "Unexpected end of input"
parseAtom (token:_) = Left $ ParseError $ "Unexpected token: " ++ show token

-- Parse function calls
parseFunctionCall :: String -> [Token] -> Either EvalError (Expr, [Token])
parseFunctionCall funcName tokens = do
    (args, rest) <- parseArguments tokens []
    case rest of
        TRParen:rest2 -> Right (Call funcName args, rest2)
        _ -> Left $ ParseError "Expected closing parenthesis in function call"
  where
    parseArguments (TRParen:_) acc = Right (reverse acc, TRParen:tokens)
    parseArguments tokens acc = do
        (expr, rest1) <- parseExpr tokens
        case rest1 of
            TComma:rest2 -> parseArguments rest2 (expr:acc)
            _ -> Right (reverse (expr:acc), rest1)

-- Top-level parse function
parseExpression :: String -> Either EvalError Expr
parseExpression input = do
    let tokens = tokenize input
    (expr, rest) <- parseExpr tokens
    case rest of
        [TEOF] -> Right expr
        _ -> Left $ ParseError "Unexpected tokens after expression"

-- Expression evaluation with environment
evaluate :: EvalEnv -> Expr -> EvalResult
evaluate _ (Num n) = Right n

evaluate env (Var name) = 
    case Map.lookup name (variables env) of
        Just value -> Right value
        Nothing -> Left $ UnknownVariable name

evaluate env (Add e1 e2) = do
    v1 <- evaluate env e1
    v2 <- evaluate env e2
    Right (v1 + v2)

evaluate env (Sub e1 e2) = do
    v1 <- evaluate env e1
    v2 <- evaluate env e2
    Right (v1 - v2)

evaluate env (Mul e1 e2) = do
    v1 <- evaluate env e1
    v2 <- evaluate env e2
    Right (v1 * v2)

evaluate env (Div e1 e2) = do
    v1 <- evaluate env e1
    v2 <- evaluate env e2
    if v2 == 0
        then Left DivisionByZero
        else Right (v1 / v2)

evaluate env (Pow e1 e2) = do
    v1 <- evaluate env e1
    v2 <- evaluate env e2
    Right (v1 ** v2)

evaluate env (Neg e) = do
    v <- evaluate env e
    Right (-v)

evaluate env (Call funcName args) = do
    case Map.lookup funcName (functions env) of
        Nothing -> Left $ UnknownFunction funcName
        Just (params, body) -> do
            when (length args /= length params) $
                Left $ ArityError funcName (length params) (length args)
            argValues <- mapM (evaluate env) args
            let newVars = Map.fromList (zip params argValues)
            let newEnv = env { variables = Map.union newVars (variables env) }
            evaluate newEnv body

-- Expression optimization using pattern matching
optimize :: Expr -> Expr
optimize expr = case expr of
    -- Arithmetic identities
    Add (Num 0) e -> optimize e
    Add e (Num 0) -> optimize e
    Sub e (Num 0) -> optimize e
    Sub (Num 0) e -> optimize (Neg e)
    Mul (Num 0) _ -> Num 0
    Mul _ (Num 0) -> Num 0
    Mul (Num 1) e -> optimize e
    Mul e (Num 1) -> optimize e
    Div e (Num 1) -> optimize e
    Pow _ (Num 0) -> Num 1
    Pow e (Num 1) -> optimize e
    Pow (Num 1) _ -> Num 1
    Neg (Neg e) -> optimize e
    
    -- Constant folding
    Add (Num a) (Num b) -> Num (a + b)
    Sub (Num a) (Num b) -> Num (a - b)
    Mul (Num a) (Num b) -> Num (a * b)
    Div (Num a) (Num b) | b /= 0 -> Num (a / b)
    Pow (Num a) (Num b) -> Num (a ** b)
    Neg (Num a) -> Num (-a)
    
    -- Recursive optimization
    Add e1 e2 -> Add (optimize e1) (optimize e2)
    Sub e1 e2 -> Sub (optimize e1) (optimize e2)
    Mul e1 e2 -> Mul (optimize e1) (optimize e2)
    Div e1 e2 -> Div (optimize e1) (optimize e2)
    Pow e1 e2 -> Pow (optimize e1) (optimize e2)
    Neg e -> Neg (optimize e)
    Call name args -> Call name (map optimize args)
    
    -- Base cases
    _ -> expr

-- Expression pretty printing
prettyPrint :: Expr -> String
prettyPrint = prettyPrintWithPrec 0
  where
    prettyPrintWithPrec :: Int -> Expr -> String
    prettyPrintWithPrec _ (Num n) 
        | fromIntegral (round n) == n = show (round n)
        | otherwise = show n
    prettyPrintWithPrec _ (Var name) = name
    prettyPrintWithPrec prec (Add e1 e2) = 
        parenthesizeIf (prec > 1) $ prettyPrintWithPrec 1 e1 ++ " + " ++ prettyPrintWithPrec 1 e2
    prettyPrintWithPrec prec (Sub e1 e2) = 
        parenthesizeIf (prec > 1) $ prettyPrintWithPrec 1 e1 ++ " - " ++ prettyPrintWithPrec 2 e2
    prettyPrintWithPrec prec (Mul e1 e2) = 
        parenthesizeIf (prec > 2) $ prettyPrintWithPrec 2 e1 ++ " * " ++ prettyPrintWithPrec 2 e2
    prettyPrintWithPrec prec (Div e1 e2) = 
        parenthesizeIf (prec > 2) $ prettyPrintWithPrec 2 e1 ++ " / " ++ prettyPrintWithPrec 3 e2
    prettyPrintWithPrec prec (Pow e1 e2) = 
        parenthesizeIf (prec > 3) $ prettyPrintWithPrec 4 e1 ++ "^" ++ prettyPrintWithPrec 3 e2
    prettyPrintWithPrec prec (Neg e) = 
        parenthesizeIf (prec > 4) $ "-" ++ prettyPrintWithPrec 4 e
    prettyPrintWithPrec _ (Call name args) = 
        name ++ "(" ++ intercalate ", " (map prettyPrint args) ++ ")"
    
    parenthesizeIf True s = "(" ++ s ++ ")"
    parenthesizeIf False s = s

-- Symbolic differentiation
differentiate :: String -> Expr -> Expr
differentiate var expr = case expr of
    Num _ -> Num 0
    Var v | v == var -> Num 1
          | otherwise -> Num 0
    Add e1 e2 -> Add (differentiate var e1) (differentiate var e2)
    Sub e1 e2 -> Sub (differentiate var e1) (differentiate var e2)
    Mul e1 e2 -> Add (Mul (differentiate var e1) e2) (Mul e1 (differentiate var e2))
    Div e1 e2 -> 
        let de1 = differentiate var e1
            de2 = differentiate var e2
        in Div (Sub (Mul de1 e2) (Mul e1 de2)) (Pow e2 (Num 2))
    Pow e1 (Num n) -> Mul (Mul (Num n) (Pow e1 (Num (n-1)))) (differentiate var e1)
    Pow e1 e2 -> -- General power rule (more complex)
        Mul (Pow e1 e2) 
            (Add (Mul (differentiate var e2) (Call "ln" [e1]))
                 (Mul e2 (Div (differentiate var e1) e1)))
    Neg e -> Neg (differentiate var e)
    Call "sin" [e] -> Mul (Call "cos" [e]) (differentiate var e)
    Call "cos" [e] -> Neg (Mul (Call "sin" [e]) (differentiate var e))
    Call "ln" [e] -> Div (differentiate var e) e
    Call "exp" [e] -> Mul (Call "exp" [e]) (differentiate var e)
    Call _ _ -> Num 0  -- Unknown function, assume constant

-- Default environment with built-in functions
defaultEnv :: EvalEnv
defaultEnv = EvalEnv
    { variables = Map.fromList [("pi", pi), ("e", exp 1)]
    , functions = Map.fromList
        [ ("sin", (["x"], Call "sin" [Var "x"]))
        , ("cos", (["x"], Call "cos" [Var "x"]))
        , ("tan", (["x"], Call "tan" [Var "x"]))
        , ("ln", (["x"], Call "ln" [Var "x"]))
        , ("log", (["x"], Call "log" [Var "x"]))
        , ("exp", (["x"], Call "exp" [Var "x"]))
        , ("sqrt", (["x"], Call "sqrt" [Var "x"]))
        , ("abs", (["x"], Call "abs" [Var "x"]))
        , ("max", (["x", "y"], Call "max" [Var "x", Var "y"]))
        , ("min", (["x", "y"], Call "min" [Var "x", Var "y"]))
        ]
    }

-- Built-in function evaluation
evaluateBuiltin :: String -> [Double] -> EvalResult
evaluateBuiltin "sin" [x] = Right (sin x)
evaluateBuiltin "cos" [x] = Right (cos x)
evaluateBuiltin "tan" [x] = Right (tan x)
evaluateBuiltin "ln" [x] = if x > 0 then Right (log x) else Left $ TypeError "ln: negative argument"
evaluateBuiltin "log" [x] = if x > 0 then Right (log x / log 10) else Left $ TypeError "log: negative argument"
evaluateBuiltin "exp" [x] = Right (exp x)
evaluateBuiltin "sqrt" [x] = if x >= 0 then Right (sqrt x) else Left $ TypeError "sqrt: negative argument"
evaluateBuiltin "abs" [x] = Right (abs x)
evaluateBuiltin "max" [x, y] = Right (max x y)
evaluateBuiltin "min" [x, y] = Right (min x y)
evaluateBuiltin name _ = Left $ UnknownFunction name

-- Enhanced evaluation with built-in functions
evaluateWithBuiltins :: EvalEnv -> Expr -> EvalResult
evaluateWithBuiltins env expr = case expr of
    Call funcName args -> 
        case Map.lookup funcName (functions env) of
            Just _ -> evaluate env expr  -- User-defined function
            Nothing -> do  -- Try built-in function
                argValues <- mapM (evaluateWithBuiltins env) args
                evaluateBuiltin funcName argValues
    _ -> evaluate env expr

-- Interactive REPL
data Command 
    = Eval String
    | Define String String  -- variable = expression
    | DefineFunc String [String] String  -- function(params) = expression
    | Differentiate String String  -- diff variable expression
    | Optimize String
    | Help
    | Quit
    deriving (Show)

parseCommand :: String -> Either String Command
parseCommand input = 
    let trimmed = dropWhile isSpace input
    in case words trimmed of
        [] -> Left "Empty command"
        ["help"] -> Right Help
        ["quit"] -> Right Quit
        ["exit"] -> Right Quit
        ("diff":var:rest) -> Right $ Differentiate var (unwords rest)
        ("optimize":rest) -> Right $ Optimize (unwords rest)
        _ -> case break (== '=') trimmed of
            (left, '=':right) -> 
                let leftTrim = reverse . dropWhile isSpace . reverse . dropWhile isSpace $ left
                    rightTrim = dropWhile isSpace right
                in if '(' `elem` leftTrim
                   then parseFunctionDef leftTrim rightTrim
                   else Right $ Define leftTrim rightTrim
            _ -> Right $ Eval trimmed
  where
    parseFunctionDef left right = 
        case break (== '(') left of
            (funcName, '(':rest) ->
                case break (== ')') rest of
                    (paramStr, ')':_) ->
                        let params = map (dropWhile isSpace . reverse . dropWhile isSpace . reverse) 
                                   $ words $ map (\c -> if c == ',' then ' ' else c) paramStr
                        in Right $ DefineFunc (dropWhile isSpace funcName) params right
                    _ -> Left "Invalid function definition: missing closing parenthesis"
            _ -> Left "Invalid function definition"

repl :: EvalEnv -> IO ()
repl env = do
    putStr "calc> "
    input <- getLine
    case parseCommand input of
        Left err -> do
            putStrLn $ "Parse error: " ++ err
            repl env
        Right Help -> do
            putStrLn "Commands:"
            putStrLn "  <expression>           - Evaluate expression"
            putStrLn "  <var> = <expression>   - Define variable"
            putStrLn "  <func>(<params>) = <expr> - Define function"
            putStrLn "  diff <var> <expression> - Differentiate"
            putStrLn "  optimize <expression>   - Optimize expression"
            putStrLn "  help                   - Show this help"
            putStrLn "  quit                   - Exit"
            repl env
        Right Quit -> putStrLn "Goodbye!"
        Right (Eval exprStr) -> do
            case parseExpression exprStr of
                Left err -> putStrLn $ "Parse error: " ++ show err
                Right expr -> 
                    case evaluateWithBuiltins env expr of
                        Left err -> putStrLn $ "Evaluation error: " ++ show err
                        Right result -> putStrLn $ show result
            repl env
        Right (Define var exprStr) -> do
            case parseExpression exprStr of
                Left err -> do
                    putStrLn $ "Parse error: " ++ show err
                    repl env
                Right expr ->
                    case evaluateWithBuiltins env expr of
                        Left err -> do
                            putStrLn $ "Evaluation error: " ++ show err
                            repl env
                        Right value -> do
                            putStrLn $ var ++ " = " ++ show value
                            let newEnv = env { variables = Map.insert var value (variables env) }
                            repl newEnv
        Right (DefineFunc funcName params exprStr) -> do
            case parseExpression exprStr of
                Left err -> do
                    putStrLn $ "Parse error: " ++ show err
                    repl env
                Right expr -> do
                    putStrLn $ "Defined function: " ++ funcName ++ "(" ++ intercalate ", " params ++ ")"
                    let newEnv = env { functions = Map.insert funcName (params, expr) (functions env) }
                    repl newEnv
        Right (Differentiate var exprStr) -> do
            case parseExpression exprStr of
                Left err -> putStrLn $ "Parse error: " ++ show err
                Right expr -> do
                    let derivative = optimize $ differentiate var expr
                    putStrLn $ "d/d" ++ var ++ " " ++ prettyPrint expr ++ " = " ++ prettyPrint derivative
            repl env
        Right (Optimize exprStr) -> do
            case parseExpression exprStr of
                Left err -> putStrLn $ "Parse error: " ++ show err
                Right expr -> do
                    let optimized = optimize expr
                    putStrLn $ "Original:  " ++ prettyPrint expr
                    putStrLn $ "Optimized: " ++ prettyPrint optimized
            repl env

-- Main function with examples
main :: IO ()
main = do
    putStrLn "=== Expression Evaluator Demo ==="
    putStrLn ""
    
    -- Example expressions
    let examples = 
            [ "2 + 3 * 4"
            , "2^3^2"
            , "(1 + 2) * (3 + 4)"
            , "sin(pi/2) + cos(0)"
            , "ln(e) + sqrt(16)"
            ]
    
    putStrLn "Example evaluations:"
    mapM_ (\expr -> do
        case parseExpression expr of
            Left err -> putStrLn $ expr ++ " -> Error: " ++ show err
            Right parsed -> 
                case evaluateWithBuiltins defaultEnv parsed of
                    Left err -> putStrLn $ expr ++ " -> Error: " ++ show err
                    Right result -> putStrLn $ expr ++ " = " ++ show result
        ) examples
    
    putStrLn ""
    putStrLn "Optimization examples:"
    let optimizeExamples = ["x + 0", "x * 1", "x^1", "0 * y", "2 + 3", "-(-(x))"]
    mapM_ (\expr -> do
        case parseExpression expr of
            Left err -> putStrLn $ expr ++ " -> Parse Error"
            Right parsed -> do
                let optimized = optimize parsed
                putStrLn $ expr ++ " -> " ++ prettyPrint optimized
        ) optimizeExamples
    
    putStrLn ""
    putStrLn "Differentiation examples:"
    let diffExamples = [("x", "x^2"), ("x", "sin(x)"), ("x", "x^3 + 2*x + 1")]
    mapM_ (\(var, expr) -> do
        case parseExpression expr of
            Left err -> putStrLn $ "d/d" ++ var ++ " " ++ expr ++ " -> Parse Error"
            Right parsed -> do
                let derivative = optimize $ differentiate var parsed
                putStrLn $ "d/d" ++ var ++ " " ++ expr ++ " = " ++ prettyPrint derivative
        ) diffExamples
    
    putStrLn ""
    putStrLn "Starting interactive REPL (type 'help' for commands):"
    repl defaultEnv