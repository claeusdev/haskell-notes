# Introduction to Haskell - Examples

## Basic Hello World

### Simple Hello World
```haskell
-- hello.hs
main :: IO ()
main = putStrLn "Hello, World!"
```

To run:
```bash
ghc hello.hs
./hello
```

### Interactive Hello World
```haskell
-- interactive_hello.hs
main :: IO ()
main = do
    putStrLn "What's your name?"
    name <- getLine
    putStrLn ("Hello, " ++ name ++ "!")
```

## Basic Expressions and Functions

### Simple Expressions
```haskell
-- In GHCi:
-- Simple arithmetic
5 + 3        -- 8
10 * 2       -- 20
15 / 3       -- 5.0
2 ^ 3        -- 8

-- Boolean operations
True && False  -- False
True || False  -- True
not True       -- False

-- Comparisons
5 == 5         -- True
10 > 5         -- True
"hello" == "hello"  -- True
```

### Basic Functions
```haskell
-- simple_functions.hs

-- Function to double a number
double :: Int -> Int
double x = x * 2

-- Function to check if number is even
isEven :: Int -> Bool
isEven n = n `mod` 2 == 0

-- Function to get absolute value
absolute :: Int -> Int
absolute n = if n < 0 then -n else n

-- Function with multiple parameters
add :: Int -> Int -> Int
add x y = x + y

-- Using functions
main :: IO ()
main = do
    print (double 5)        -- 10
    print (isEven 4)        -- True
    print (absolute (-3))   -- 3
    print (add 10 20)       -- 30
```

## Lists and Basic Operations

### List Examples
```haskell
-- lists.hs

-- Creating lists
numbers :: [Int]
numbers = [1, 2, 3, 4, 5]

names :: [String]
names = ["Alice", "Bob", "Charlie"]

-- List operations
firstNumber = head numbers      -- 1
restNumbers = tail numbers      -- [2,3,4,5]
listSize = length numbers       -- 5

-- Checking if list is empty
isEmpty = null []               -- True
notEmpty = null numbers         -- False

-- Concatenating lists
combined = [1, 2] ++ [3, 4]     -- [1,2,3,4]

-- Adding element to front
newList = 0 : numbers           -- [0,1,2,3,4,5]

main :: IO ()
main = do
    print numbers
    print firstNumber
    print listSize
    print combined
```

## Type Signatures and Inference

### Explicit Type Signatures
```haskell
-- types.hs

-- Explicit type signatures
greet :: String -> String
greet name = "Hello, " ++ name

multiply :: Int -> Int -> Int
multiply x y = x * y

-- Polymorphic function
identity :: a -> a
identity x = x

-- Function with multiple type constraints
compare' :: (Ord a, Show a) => a -> a -> String
compare' x y 
    | x > y = show x ++ " is greater than " ++ show y
    | x < y = show x ++ " is less than " ++ show y
    | otherwise = show x ++ " equals " ++ show y

main :: IO ()
main = do
    putStrLn (greet "World")
    print (multiply 6 7)
    print (identity 42)
    putStrLn (compare' 10 5)
```

## Basic Pattern Matching

### Simple Pattern Matching
```haskell
-- patterns.hs

-- Pattern matching on numbers
describe :: Int -> String
describe 0 = "zero"
describe 1 = "one"
describe 2 = "two"
describe n = "some number: " ++ show n

-- Pattern matching on booleans
boolToString :: Bool -> String
boolToString True = "yes"
boolToString False = "no"

-- Pattern matching on lists
listDescription :: [a] -> String
listDescription [] = "empty list"
listDescription [x] = "single element list"
listDescription [x, y] = "two element list"
listDescription _ = "longer list"

main :: IO ()
main = do
    putStrLn (describe 0)        -- "zero"
    putStrLn (describe 5)        -- "some number: 5"
    putStrLn (boolToString True) -- "yes"
    putStrLn (listDescription []) -- "empty list"
    putStrLn (listDescription [1, 2, 3]) -- "longer list"
```

## Simple Recursion

### Basic Recursive Functions
```haskell
-- recursion.hs

-- Factorial function
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Fibonacci function
fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

-- Sum of list elements
sumList :: [Int] -> Int
sumList [] = 0
sumList (x:xs) = x + sumList xs

-- Length of list
listLength :: [a] -> Int
listLength [] = 0
listLength (_:xs) = 1 + listLength xs

main :: IO ()
main = do
    print (factorial 5)         -- 120
    print (fibonacci 8)         -- 21
    print (sumList [1,2,3,4])   -- 10
    print (listLength ['a','b','c']) -- 3
```

## Guards and Where Clauses

### Using Guards
```haskell
-- guards.hs

-- BMI calculator with guards
bmiTell :: Float -> Float -> String
bmiTell weight height
    | bmi <= 18.5 = "You're underweight!"
    | bmi <= 25.0 = "You're normal weight!"
    | bmi <= 30.0 = "You're overweight!"
    | otherwise = "You're obese!"
    where bmi = weight / height ^ 2

-- Grade calculator
gradeFromScore :: Int -> Char
gradeFromScore score
    | score >= 90 = 'A'
    | score >= 80 = 'B'
    | score >= 70 = 'C'
    | score >= 60 = 'D'
    | otherwise = 'F'

main :: IO ()
main = do
    putStrLn (bmiTell 70 1.75)  -- "You're normal weight!"
    print (gradeFromScore 85)   -- 'B'
```

## Let Expressions

### Local Bindings with Let
```haskell
-- let_expressions.hs

-- Using let in expressions
cylinder :: Float -> Float -> Float
cylinder r h = 
    let sideArea = 2 * pi * r * h
        topArea = pi * r ^ 2
    in sideArea + 2 * topArea

-- Let in list comprehensions
squareAndDouble :: [Int] -> [Int]
squareAndDouble xs = [let square = x * x in square * 2 | x <- xs]

-- Complex calculation with let
complexCalculation :: Float -> Float
complexCalculation x = 
    let a = x * 2
        b = a + 10
        c = b / 3
    in a + b + c

main :: IO ()
main = do
    print (cylinder 2 3)
    print (squareAndDouble [1,2,3,4])
    print (complexCalculation 5)
```

## List Comprehensions

### Basic List Comprehensions
```haskell
-- list_comprehensions.hs

-- Simple list comprehension
squares :: [Int] -> [Int]
squares xs = [x * x | x <- xs]

-- With condition
evens :: [Int] -> [Int]
evens xs = [x | x <- xs, even x]

-- Multiple generators
pairs :: [Int] -> [Int] -> [(Int, Int)]
pairs xs ys = [(x, y) | x <- xs, y <- ys]

-- Pythagorean triples
pythagoreanTriples :: Int -> [(Int, Int, Int)]
pythagoreanTriples n = [(a, b, c) | c <- [1..n], 
                                   b <- [1..c], 
                                   a <- [1..b], 
                                   a^2 + b^2 == c^2]

main :: IO ()
main = do
    print (squares [1,2,3,4])           -- [1,4,9,16]
    print (evens [1,2,3,4,5,6])         -- [2,4,6]
    print (pairs [1,2] ['a','b'])       -- [(1,'a'),(1,'b'),(2,'a'),(2,'b')]
    print (pythagoreanTriples 10)       -- [(3,4,5),(6,8,10)]
```

## Higher-Order Function Basics

### Map, Filter, and Fold
```haskell
-- higher_order.hs

-- Using map
doubleAll :: [Int] -> [Int]
doubleAll xs = map (*2) xs

-- Using filter
positiveNumbers :: [Int] -> [Int]
positiveNumbers xs = filter (>0) xs

-- Using foldr
sumAll :: [Int] -> Int
sumAll xs = foldr (+) 0 xs

-- Using foldl
productAll :: [Int] -> Int
productAll xs = foldl (*) 1 xs

-- Combining higher-order functions
processNumbers :: [Int] -> Int
processNumbers xs = 
    foldr (+) 0 $           -- sum
    map (*2) $              -- double each
    filter (>0) xs          -- keep positive

main :: IO ()
main = do
    print (doubleAll [1,2,3,4])         -- [2,4,6,8]
    print (positiveNumbers [-2,-1,0,1,2]) -- [1,2]
    print (sumAll [1,2,3,4])            -- 10
    print (productAll [1,2,3,4])        -- 24
    print (processNumbers [-1,2,-3,4])  -- 12
```

## Error Handling with Maybe

### Basic Maybe Usage
```haskell
-- maybe_examples.hs

-- Safe division
safeDivide :: Float -> Float -> Maybe Float
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

-- Safe head function
safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x

-- Working with Maybe values
processResult :: Maybe Float -> String
processResult Nothing = "Error: Invalid operation"
processResult (Just x) = "Result: " ++ show x

main :: IO ()
main = do
    putStrLn (processResult (safeDivide 10 2))  -- "Result: 5.0"
    putStrLn (processResult (safeDivide 10 0))  -- "Error: Invalid operation"
    print (safeHead [1,2,3])                   -- Just 1
    print (safeHead ([] :: [Int]))              -- Nothing
```

## Running the Examples

To run any of these examples:

1. Save the code to a `.hs` file
2. Compile with GHC: `ghc filename.hs`
3. Run the executable: `./filename`

Or use GHCi for interactive testing:
```bash
ghci filename.hs
```

Then you can test individual functions:
```haskell
*Main> factorial 5
120
*Main> fibonacci 8
21
```