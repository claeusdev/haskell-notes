# Pattern Matching and Recursion - Examples

## Basic Pattern Matching Examples

### Simple Value Patterns
```haskell
-- simple_patterns.hs

-- Boolean pattern matching
not' :: Bool -> Bool
not' True = False
not' False = True

-- Character pattern matching
isVowel :: Char -> Bool
isVowel 'a' = True
isVowel 'e' = True
isVowel 'i' = True
isVowel 'o' = True
isVowel 'u' = True
isVowel _ = False

-- Number pattern matching with guards
classify :: Int -> String
classify n
    | n < 0 = "negative"
    | n == 0 = "zero"
    | n > 0 && n <= 10 = "small positive"
    | otherwise = "large positive"

main :: IO ()
main = do
    print (not' True)          -- False
    print (isVowel 'a')        -- True
    print (isVowel 'x')        -- False
    print (classify 5)         -- "small positive"
    print (classify (-3))      -- "negative"
```

### List Pattern Matching
```haskell
-- list_patterns.hs

-- Basic list patterns
listLength :: [a] -> Int
listLength [] = 0
listLength (_:xs) = 1 + listLength xs

-- Multiple element patterns
firstTwo :: [a] -> Maybe (a, a)
firstTwo [] = Nothing
firstTwo [_] = Nothing
firstTwo (x:y:_) = Just (x, y)

-- Complex list patterns
describelist :: [a] -> String
describelist [] = "empty list"
describelist [_] = "singleton list"
describelist [_, _] = "list with two elements"
describelist _ = "list with many elements"

-- Pattern matching with conditions
takePositive :: [Int] -> [Int]
takePositive [] = []
takePositive (x:xs)
    | x > 0 = x : takePositive xs
    | otherwise = takePositive xs

-- Nested list patterns
sumPairs :: [(Int, Int)] -> [Int]
sumPairs [] = []
sumPairs ((x, y):rest) = (x + y) : sumPairs rest

main :: IO ()
main = do
    print (listLength [1,2,3,4])           -- 4
    print (firstTwo [1,2,3])               -- Just (1,2)
    print (firstTwo [1])                   -- Nothing
    print (describelist [1,2,3])           -- "list with many elements"
    print (takePositive [1,-2,3,-4,5])     -- [1,3,5]
    print (sumPairs [(1,2), (3,4), (5,6)]) -- [3,7,11]
```

### As-Patterns and Wildcards
```haskell
-- as_patterns.hs

-- As-patterns (@) - capture both whole and parts
duplicate :: [a] -> [a]
duplicate [] = []
duplicate all@(x:xs) = x : all  -- x:x:xs equivalent

-- More complex as-pattern
tails :: [a] -> [[a]]
tails [] = [[]]
tails all@(_:xs) = all : tails xs

-- Wildcard patterns
count :: [a] -> Int
count [] = 0
count (_:xs) = 1 + count xs

-- Extract specific elements
third :: [a] -> Maybe a
third [] = Nothing
third [_] = Nothing
third [_, _] = Nothing
third (_:_:x:_) = Just x

-- Pattern matching with records
data Person = Person { name :: String, age :: Int, city :: String }

greet :: Person -> String
greet (Person n a c) = "Hello " ++ n ++ " from " ++ c ++ ", age " ++ show a

-- Using wildcards in records
isAdult :: Person -> Bool
isAdult (Person _ age _) = age >= 18

main :: IO ()
main = do
    print (duplicate [1,2,3])              -- [1,1,2,3]
    print (tails [1,2,3])                  -- [[1,2,3],[2,3],[3],[]]
    print (count [1,2,3,4,5])              -- 5
    print (third [1,2,3,4])                -- Just 3
    print (third [1,2])                    -- Nothing
    
    let person = Person "Alice" 25 "NYC"
    putStrLn (greet person)                -- "Hello Alice from NYC, age 25"
    print (isAdult person)                 -- True
```

## Recursion Examples

### Linear Recursion
```haskell
-- linear_recursion.hs

-- Basic linear recursion
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Linear recursion with accumulator (tail recursion)
factorialTail :: Int -> Int
factorialTail n = factorialHelper n 1
  where
    factorialHelper 0 acc = acc
    factorialHelper n acc = factorialHelper (n - 1) (n * acc)

-- String reversal
reverseString :: String -> String
reverseString [] = []
reverseString (x:xs) = reverseString xs ++ [x]

-- Tail-recursive version
reverseStringTail :: String -> String
reverseStringTail s = reverseHelper s []
  where
    reverseHelper [] acc = acc
    reverseHelper (x:xs) acc = reverseHelper xs (x:acc)

-- List operations
myLength :: [a] -> Int
myLength [] = 0
myLength (_:xs) = 1 + myLength xs

mySum :: [Int] -> Int
mySum [] = 0
mySum (x:xs) = x + mySum xs

-- More complex example: quick sort
quickSort :: Ord a => [a] -> [a]
quickSort [] = []
quickSort (pivot:rest) = 
    quickSort smaller ++ [pivot] ++ quickSort larger
  where
    smaller = [x | x <- rest, x <= pivot]
    larger = [x | x <- rest, x > pivot]

main :: IO ()
main = do
    print (factorial 5)                    -- 120
    print (factorialTail 5)                -- 120
    putStrLn (reverseString "hello")       -- "olleh"
    putStrLn (reverseStringTail "hello")   -- "olleh"
    print (myLength [1,2,3,4])             -- 4
    print (mySum [1,2,3,4])                -- 10
    print (quickSort [3,1,4,1,5,9,2,6])    -- [1,1,2,3,4,5,6,9]
```

### Tree Recursion
```haskell
-- tree_recursion.hs

-- Fibonacci with tree recursion (inefficient)
fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

-- Optimized with memoization
fibonacciMemo :: Int -> Int
fibonacciMemo = (map fib [0..] !!)
  where
    fib 0 = 0
    fib 1 = 1
    fib n = fibonacciMemo (n - 1) + fibonacciMemo (n - 2)

-- Binary tree operations
data Tree a = Empty | Node a (Tree a) (Tree a)
    deriving (Show, Eq)

-- Tree traversals
inorderTraversal :: Tree a -> [a]
inorderTraversal Empty = []
inorderTraversal (Node x left right) = 
    inorderTraversal left ++ [x] ++ inorderTraversal right

preorderTraversal :: Tree a -> [a]
preorderTraversal Empty = []
preorderTraversal (Node x left right) = 
    [x] ++ preorderTraversal left ++ preorderTraversal right

postorderTraversal :: Tree a -> [a]
postorderTraversal Empty = []
postorderTraversal (Node x left right) = 
    postorderTraversal left ++ postorderTraversal right ++ [x]

-- Tree operations
treeHeight :: Tree a -> Int
treeHeight Empty = 0
treeHeight (Node _ left right) = 
    1 + max (treeHeight left) (treeHeight right)

treeSize :: Tree a -> Int
treeSize Empty = 0
treeSize (Node _ left right) = 
    1 + treeSize left + treeSize right

-- Insert into binary search tree
insertBST :: Ord a => a -> Tree a -> Tree a
insertBST x Empty = Node x Empty Empty
insertBST x (Node y left right)
    | x <= y = Node y (insertBST x left) right
    | otherwise = Node y left (insertBST x right)

-- Search in binary search tree
searchBST :: Ord a => a -> Tree a -> Bool
searchBST _ Empty = False
searchBST x (Node y left right)
    | x == y = True
    | x < y = searchBST x left
    | otherwise = searchBST x right

-- Sample tree
sampleTree :: Tree Int
sampleTree = Node 5 
                (Node 3 (Node 1 Empty Empty) (Node 4 Empty Empty))
                (Node 8 (Node 6 Empty Empty) (Node 9 Empty Empty))

main :: IO ()
main = do
    print (fibonacci 10)                    -- 55 (slow for large numbers)
    print (fibonacciMemo 30)                -- 832040 (fast)
    
    print (inorderTraversal sampleTree)     -- [1,3,4,5,6,8,9]
    print (preorderTraversal sampleTree)    -- [5,3,1,4,8,6,9]
    print (postorderTraversal sampleTree)   -- [1,4,3,6,9,8,5]
    
    print (treeHeight sampleTree)           -- 3
    print (treeSize sampleTree)             -- 7
    
    let newTree = insertBST 7 sampleTree
    print (inorderTraversal newTree)        -- [1,3,4,5,6,7,8,9]
    print (searchBST 6 sampleTree)          -- True
    print (searchBST 10 sampleTree)         -- False
```

### Mutual Recursion
```haskell
-- mutual_recursion.hs

-- Classic mutual recursion example
isEven :: Int -> Bool
isEven 0 = True
isEven n = isOdd (n - 1)

isOdd :: Int -> Bool
isOdd 0 = False
isOdd n = isEven (n - 1)

-- Parser example with mutual recursion
data Expr = Num Int | Add Expr Expr | Mul Expr Expr
    deriving (Show, Eq)

-- Mutually recursive parser functions
parseExpr :: String -> (Expr, String)
parseExpr input = parseTerm input

parseTerm :: String -> (Expr, String)
parseTerm input = 
    let (left, rest1) = parseFactor input
    in case rest1 of
        ('+':rest2) -> let (right, rest3) = parseTerm rest2
                       in (Add left right, rest3)
        _ -> (left, rest1)

parseFactor :: String -> (Expr, String)
parseFactor input =
    let (left, rest1) = parseNumber input
    in case rest1 of
        ('*':rest2) -> let (right, rest3) = parseFactor rest2
                       in (Mul left right, rest3)
        _ -> (left, rest1)

parseNumber :: String -> (Expr, String)
parseNumber input = 
    let (numStr, rest) = span (`elem` "0123456789") input
    in (Num (read numStr), rest)

-- Evaluate expression
evalExpr :: Expr -> Int
evalExpr (Num n) = n
evalExpr (Add e1 e2) = evalExpr e1 + evalExpr e2
evalExpr (Mul e1 e2) = evalExpr e1 * evalExpr e2

-- Forest traversal with mutual recursion
data Tree a = Node a [Tree a]
    deriving (Show, Eq)

type Forest a = [Tree a]

-- Mutually recursive traversal
traverseForest :: Forest a -> [a]
traverseForest [] = []
traverseForest (tree:rest) = traverseTree tree ++ traverseForest rest

traverseTree :: Tree a -> [a]
traverseTree (Node x children) = x : traverseForest children

sampleForest :: Forest Int
sampleForest = [Node 1 [Node 2 [], Node 3 [Node 4 []]], Node 5 []]

main :: IO ()
main = do
    print (isEven 4)                        -- True
    print (isOdd 4)                         -- False
    print (isEven 7)                        -- False
    print (isOdd 7)                         -- True
    
    let (expr, _) = parseExpr "3+4*5"
    print expr                              -- Add (Num 3) (Mul (Num 4) (Num 5))
    print (evalExpr expr)                   -- 23
    
    print (traverseForest sampleForest)     -- [1,2,3,4,5]
```

## Advanced Pattern Matching

### Guards and Where Clauses
```haskell
-- guards_where.hs

-- Complex guards with where clauses
quadraticRoots :: Double -> Double -> Double -> String
quadraticRoots a b c
    | discriminant > 0 = "Two real roots: " ++ show root1 ++ " and " ++ show root2
    | discriminant == 0 = "One root: " ++ show (-b / (2 * a))
    | otherwise = "Complex roots"
  where
    discriminant = b * b - 4 * a * c
    root1 = (-b + sqrt discriminant) / (2 * a)
    root2 = (-b - sqrt discriminant) / (2 * a)

-- Nested where clauses
analyzeTriangle :: Double -> Double -> Double -> String
analyzeTriangle a b c
    | not isValidTriangle = "Not a valid triangle"
    | isEquilateral = "Equilateral triangle"
    | isIsosceles = "Isosceles triangle"
    | otherwise = "Scalene triangle"
  where
    isValidTriangle = a + b > c && a + c > b && b + c > a
    isEquilateral = a == b && b == c
    isIsosceles = a == b || b == c || a == c

-- Guards with pattern matching
describeList :: [a] -> String
describeList xs
    | null xs = "Empty list"
    | length xs == 1 = "Singleton: " ++ show (length xs) ++ " element"
    | length xs <= 5 = "Short list: " ++ show (length xs) ++ " elements"
    | otherwise = "Long list: " ++ show (length xs) ++ " elements"

-- Complex pattern matching with guards
processGrades :: [(String, Int)] -> [(String, String, Bool)]
processGrades [] = []
processGrades ((name, score):rest) = (name, grade, passing) : processGrades rest
  where
    (grade, passing)
        | score >= 90 = ("A", True)
        | score >= 80 = ("B", True)
        | score >= 70 = ("C", True)
        | score >= 60 = ("D", True)
        | otherwise = ("F", False)

main :: IO ()
main = do
    putStrLn (quadraticRoots 1 (-5) 6)      -- "Two real roots: 3.0 and 2.0"
    putStrLn (analyzeTriangle 3 3 3)        -- "Equilateral triangle"
    putStrLn (analyzeTriangle 3 3 4)        -- "Isosceles triangle"
    putStrLn (analyzeTriangle 3 4 5)        -- "Scalene triangle"
    putStrLn (describeList [1,2,3])         -- "Short list: 3 elements"
    print (processGrades [("Alice", 95), ("Bob", 75), ("Charlie", 55)])
```

### Case Expressions
```haskell
-- case_expressions.hs

-- Basic case expressions
dayOfWeek :: Int -> String
dayOfWeek n = case n of
    1 -> "Monday"
    2 -> "Tuesday"
    3 -> "Wednesday"
    4 -> "Thursday"
    5 -> "Friday"
    6 -> "Saturday"
    7 -> "Sunday"
    _ -> "Invalid day"

-- Case with complex patterns
processEither :: Either String Int -> String
processEither value = case value of
    Left err -> "Error: " ++ err
    Right n | n > 0 -> "Positive: " ++ show n
            | n < 0 -> "Negative: " ++ show n
            | otherwise -> "Zero"

-- Nested case expressions
evaluateExpr :: Expr -> Either String Int
evaluateExpr expr = case expr of
    Num n -> Right n
    Add e1 e2 -> case (evaluateExpr e1, evaluateExpr e2) of
        (Right v1, Right v2) -> Right (v1 + v2)
        (Left err, _) -> Left err
        (_, Left err) -> Left err
    Mul e1 e2 -> case (evaluateExpr e1, evaluateExpr e2) of
        (Right v1, Right v2) -> Right (v1 * v2)
        (Left err, _) -> Left err
        (_, Left err) -> Left err

-- Case in where clauses
fibonacci' :: Int -> Int
fibonacci' n = case n of
    0 -> 0
    1 -> 1
    _ -> fib (n-1) + fib (n-2)
  where
    fib = fibonacci'

-- Complex pattern matching in case
parseCommand :: String -> String
parseCommand input = case words input of
    [] -> "Empty command"
    ["quit"] -> "Goodbye!"
    ["help"] -> "Available commands: help, quit, echo <message>"
    ("echo":message) -> "Echo: " ++ unwords message
    [unknown] -> "Unknown command: " ++ unknown
    _ -> "Invalid command format"

main :: IO ()
main = do
    putStrLn (dayOfWeek 3)                  -- "Wednesday"
    putStrLn (dayOfWeek 8)                  -- "Invalid day"
    putStrLn (processEither (Right 5))      -- "Positive: 5"
    putStrLn (processEither (Left "error")) -- "Error: error"
    
    let expr = Add (Num 3) (Mul (Num 4) (Num 5))
    print (evaluateExpr expr)               -- Right 23
    
    print (fibonacci' 8)                    -- 21
    putStrLn (parseCommand "echo hello world") -- "Echo: hello world"
    putStrLn (parseCommand "help")          -- "Available commands: help, quit, echo <message>"
```

## Running the Examples

To run any of these examples:

1. Save the code to a `.hs` file
2. Compile with: `ghc filename.hs`
3. Run with: `./filename`

Or use GHCi for interactive testing:
```bash
ghci filename.hs
*Main> functionName arguments
```

These examples demonstrate the power and expressiveness of pattern matching and recursion in Haskell, showing how they enable elegant solutions to complex problems while maintaining code clarity and correctness.