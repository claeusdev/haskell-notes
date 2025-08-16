-- Calculator.hs
-- A comprehensive calculator demonstrating basic Haskell syntax and types
-- This project covers: data types, pattern matching, error handling, parsing

module Calculator where

import Data.Char (isDigit, isSpace)
import Text.Read (readMaybe)

-- Data types for the calculator

-- Operation type representing basic arithmetic operations
data Operation = Add | Subtract | Multiply | Divide | Power | Modulo
    deriving (Show, Eq)

-- Expression type for more complex calculations
data Expression 
    = Number Double
    | Binary Operation Expression Expression
    | Unary UnaryOp Expression
    deriving (Show, Eq)

-- Unary operations
data UnaryOp = Negate | Sqrt | Abs
    deriving (Show, Eq)

-- Result type for error handling
data CalcResult = Success Double | Error String
    deriving (Show, Eq)

-- Calculator state for memory operations
data CalculatorState = CalculatorState
    { memory :: Double
    , history :: [String]
    , lastResult :: Maybe Double
    } deriving (Show)

-- Initialize calculator state
initialState :: CalculatorState
initialState = CalculatorState 0.0 [] Nothing

-- Basic arithmetic functions with error handling

-- Safe division
safeDivide :: Double -> Double -> CalcResult
safeDivide _ 0 = Error "Division by zero"
safeDivide x y = Success (x / y)

-- Safe modulo
safeModulo :: Double -> Double -> CalcResult
safeModulo _ 0 = Error "Modulo by zero"
safeModulo x y = Success (fromIntegral (floor x `mod` floor y))

-- Safe square root
safeSqrt :: Double -> CalcResult
safeSqrt x
    | x < 0 = Error "Square root of negative number"
    | otherwise = Success (sqrt x)

-- Safe power operation
safePower :: Double -> Double -> CalcResult
safePower base exponent
    | base == 0 && exponent < 0 = Error "Zero to negative power"
    | base < 0 && exponent /= fromIntegral (floor exponent) = 
        Error "Negative base with non-integer exponent"
    | otherwise = Success (base ** exponent)

-- Evaluate operations
evaluateOperation :: Operation -> Double -> Double -> CalcResult
evaluateOperation Add x y = Success (x + y)
evaluateOperation Subtract x y = Success (x - y)
evaluateOperation Multiply x y = Success (x * y)
evaluateOperation Divide x y = safeDivide x y
evaluateOperation Power x y = safePower x y
evaluateOperation Modulo x y = safeModulo x y

-- Evaluate unary operations
evaluateUnaryOp :: UnaryOp -> Double -> CalcResult
evaluateUnaryOp Negate x = Success (-x)
evaluateUnaryOp Sqrt x = safeSqrt x
evaluateUnaryOp Abs x = Success (abs x)

-- Evaluate expressions
evaluateExpression :: Expression -> CalcResult
evaluateExpression (Number x) = Success x
evaluateExpression (Binary op left right) = do
    leftVal <- evaluateExpression left
    rightVal <- evaluateExpression right
    case (leftVal, rightVal) of
        (Success x, Success y) -> evaluateOperation op x y
        (Error err, _) -> Error err
        (_, Error err) -> Error err
evaluateExpression (Unary op expr) = do
    val <- evaluateExpression expr
    case val of
        Success x -> evaluateUnaryOp op x
        Error err -> Error err

-- Parsing functions

-- Parse a number from string
parseNumber :: String -> Maybe Double
parseNumber s = readMaybe (trim s)
  where
    trim = reverse . dropWhile isSpace . reverse . dropWhile isSpace

-- Parse operation from string
parseOperation :: String -> Maybe Operation
parseOperation "+" = Just Add
parseOperation "-" = Just Subtract
parseOperation "*" = Just Multiply
parseOperation "/" = Just Divide
parseOperation "^" = Just Power
parseOperation "%" = Just Modulo
parseOperation _ = Nothing

-- Parse unary operation
parseUnaryOp :: String -> Maybe UnaryOp
parseUnaryOp "neg" = Just Negate
parseUnaryOp "sqrt" = Just Sqrt
parseUnaryOp "abs" = Just Abs
parseUnaryOp _ = Nothing

-- Simple expression parser (handles basic binary operations)
parseExpression :: String -> Maybe Expression
parseExpression s = 
    case words s of
        [x] -> Number <$> parseNumber x
        [op, x] -> do
            unaryOp <- parseUnaryOp op
            val <- parseNumber x
            return $ Unary unaryOp (Number val)
        [x, op, y] -> do
            operation <- parseOperation op
            leftVal <- parseNumber x
            rightVal <- parseNumber y
            return $ Binary operation (Number leftVal) (Number rightVal)
        _ -> Nothing

-- Calculator interface functions

-- Calculate from string input
calculate :: String -> CalcResult
calculate input = 
    case parseExpression input of
        Just expr -> evaluateExpression expr
        Nothing -> Error "Invalid expression"

-- Format result for display
formatResult :: CalcResult -> String
formatResult (Success x) = 
    if x == fromIntegral (round x)
        then show (round x)
        else show x
formatResult (Error err) = "Error: " ++ err

-- Memory operations
storeInMemory :: Double -> CalculatorState -> CalculatorState
storeInMemory val state = state { memory = val }

recallFromMemory :: CalculatorState -> Double
recallFromMemory = memory

clearMemory :: CalculatorState -> CalculatorState
clearMemory state = state { memory = 0.0 }

-- Add to history
addToHistory :: String -> CalculatorState -> CalculatorState
addToHistory entry state = state { history = entry : take 9 (history state) }

-- Update last result
updateLastResult :: Double -> CalculatorState -> CalculatorState
updateLastResult result state = state { lastResult = Just result }

-- Special calculations

-- Calculate factorial
factorial :: Int -> CalcResult
factorial n
    | n < 0 = Error "Factorial of negative number"
    | n > 170 = Error "Factorial too large"
    | otherwise = Success (fromIntegral (fact n))
  where
    fact 0 = 1
    fact x = x * fact (x - 1)

-- Calculate greatest common divisor
gcd' :: Int -> Int -> Int
gcd' a 0 = abs a
gcd' a b = gcd' b (a `mod` b)

-- Calculate least common multiple
lcm' :: Int -> Int -> Int
lcm' a b = abs (a * b) `div` gcd' a b

-- Calculate percentage
percentage :: Double -> Double -> Double
percentage part whole = (part / whole) * 100

-- Advanced operations

-- Calculate compound interest
compoundInterest :: Double -> Double -> Double -> Int -> Double
compoundInterest principal rate time compounds =
    principal * (1 + rate / fromIntegral compounds) ** (fromIntegral compounds * time)

-- Convert between number bases
decimalToBinary :: Int -> String
decimalToBinary 0 = "0"
decimalToBinary n = decimalToBinary (n `div` 2) ++ show (n `mod` 2)

decimalToHex :: Int -> String
decimalToHex n
    | n < 16 = [hexDigit n]
    | otherwise = decimalToHex (n `div` 16) ++ [hexDigit (n `mod` 16)]
  where
    hexDigit x
        | x < 10 = toEnum (fromEnum '0' + x)
        | otherwise = toEnum (fromEnum 'A' + x - 10)

-- Statistical functions
mean :: [Double] -> CalcResult
mean [] = Error "Cannot calculate mean of empty list"
mean xs = Success (sum xs / fromIntegral (length xs))

median :: [Double] -> CalcResult
median [] = Error "Cannot calculate median of empty list"
median xs = 
    let sorted = quickSort xs
        len = length sorted
        mid = len `div` 2
    in if odd len
       then Success (sorted !! mid)
       else Success ((sorted !! (mid - 1) + sorted !! mid) / 2)

-- Quick sort implementation for median calculation
quickSort :: Ord a => [a] -> [a]
quickSort [] = []
quickSort (x:xs) = 
    quickSort [y | y <- xs, y <= x] ++ [x] ++ quickSort [y | y <- xs, y > x]

-- Calculator REPL (Read-Eval-Print Loop)
calculatorREPL :: CalculatorState -> IO ()
calculatorREPL state = do
    putStr "Calculator> "
    input <- getLine
    case input of
        "quit" -> putStrLn "Goodbye!"
        "exit" -> putStrLn "Goodbye!"
        "help" -> do
            showHelp
            calculatorREPL state
        "history" -> do
            showHistory state
            calculatorREPL state
        "memory" -> do
            putStrLn $ "Memory: " ++ show (memory state)
            calculatorREPL state
        "clear" -> do
            putStrLn "Memory cleared"
            calculatorREPL (clearMemory state)
        _ -> do
            let result = calculate input
            putStrLn $ formatResult result
            let newState = case result of
                    Success val -> updateLastResult val $ addToHistory 
                        (input ++ " = " ++ formatResult result) state
                    Error _ -> addToHistory 
                        (input ++ " = " ++ formatResult result) state
            calculatorREPL newState

-- Show help information
showHelp :: IO ()
showHelp = do
    putStrLn "Calculator Commands:"
    putStrLn "  Basic operations: +, -, *, /, ^, %"
    putStrLn "  Unary operations: neg, sqrt, abs"
    putStrLn "  Example: 5 + 3"
    putStrLn "  Example: sqrt 16"
    putStrLn "  Special commands:"
    putStrLn "    help    - Show this help"
    putStrLn "    history - Show calculation history"
    putStrLn "    memory  - Show memory value"
    putStrLn "    clear   - Clear memory"
    putStrLn "    quit    - Exit calculator"

-- Show calculation history
showHistory :: CalculatorState -> IO ()
showHistory state = do
    putStrLn "Calculation History:"
    if null (history state)
        then putStrLn "  No history"
        else mapM_ (\(i, entry) -> putStrLn $ "  " ++ show i ++ ". " ++ entry) 
             (zip [1..] (reverse $ history state))

-- Test functions
testCalculator :: IO ()
testCalculator = do
    putStrLn "Testing Calculator..."
    
    -- Test basic operations
    putStrLn $ "5 + 3 = " ++ formatResult (calculate "5 + 3")
    putStrLn $ "10 - 4 = " ++ formatResult (calculate "10 - 4")
    putStrLn $ "6 * 7 = " ++ formatResult (calculate "6 * 7")
    putStrLn $ "15 / 3 = " ++ formatResult (calculate "15 / 3")
    putStrLn $ "2 ^ 8 = " ++ formatResult (calculate "2 ^ 8")
    putStrLn $ "17 % 5 = " ++ formatResult (calculate "17 % 5")
    
    -- Test unary operations
    putStrLn $ "sqrt 16 = " ++ formatResult (calculate "sqrt 16")
    putStrLn $ "abs -5 = " ++ formatResult (calculate "abs -5")
    putStrLn $ "neg 10 = " ++ formatResult (calculate "neg 10")
    
    -- Test error cases
    putStrLn $ "10 / 0 = " ++ formatResult (calculate "10 / 0")
    putStrLn $ "sqrt -4 = " ++ formatResult (calculate "sqrt -4")
    
    putStrLn "All tests completed!"

-- Main function
main :: IO ()
main = do
    putStrLn "============================="
    putStrLn "  Haskell Calculator v1.0"
    putStrLn "============================="
    putStrLn "Type 'help' for commands or 'quit' to exit"
    putStrLn ""
    calculatorREPL initialState

-- Example usage:
-- ghc Calculator.hs -o calculator
-- ./calculator
-- 
-- Or in GHCi:
-- :load Calculator
-- main