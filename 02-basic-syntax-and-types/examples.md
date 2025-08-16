# Basic Syntax and Types - Examples

## Primitive Types Examples

### Numeric Types
```haskell
-- numeric_types.hs

-- Integer types
smallInt :: Int
smallInt = 42

bigInt :: Integer
bigInt = 123456789012345678901234567890

-- Floating point
singlePrecision :: Float
singlePrecision = 3.14159

doublePrecision :: Double
doublePrecision = 2.718281828459045

-- Arithmetic operations
main :: IO ()
main = do
    print (smallInt + 10)           -- 52
    print (bigInt * 2)              -- Very large number
    print (singlePrecision * 2)     -- 6.28318
    print (doublePrecision + 1)     -- 3.718281828459045
```

### Character and String Types
```haskell
-- char_string.hs

-- Character operations
letter :: Char
letter = 'H'

digit :: Char
digit = '7'

-- String operations
greeting :: String
greeting = "Hello, World!"

name :: String
name = "Haskell"

-- String functions
stringOps :: IO ()
stringOps = do
    print letter                    -- 'H'
    print (greeting ++ " " ++ name) -- "Hello, World! Haskell"
    print (length greeting)         -- 13
    print (reverse greeting)        -- "!dlroW ,olleH"
    print (take 5 greeting)         -- "Hello"
    print (drop 7 greeting)         -- "World!"

main = stringOps
```

### Boolean Operations
```haskell
-- boolean_ops.hs

-- Boolean values
isTrue :: Bool
isTrue = True

isFalse :: Bool
isFalse = False

-- Boolean operations
booleanExamples :: IO ()
booleanExamples = do
    print (isTrue && isFalse)       -- False
    print (isTrue || isFalse)       -- True
    print (not isTrue)              -- False
    print (5 > 3)                   -- True
    print (10 == 10)                -- True
    print ("hello" /= "world")      -- True

main = booleanExamples
```

## Type Signatures and Inference

### Explicit Type Signatures
```haskell
-- type_signatures.hs

-- Simple function with explicit type
double :: Int -> Int
double x = x * 2

-- Function with multiple parameters
multiply :: Int -> Int -> Int
multiply x y = x * y

-- Polymorphic function
identity :: a -> a
identity x = x

-- Function with type constraints
maximum' :: Ord a => [a] -> a
maximum' [x] = x
maximum' (x:xs) = max x (maximum' xs)

-- Function with multiple constraints
showAndSort :: (Show a, Ord a) => [a] -> String
showAndSort xs = show (sort xs)
  where
    sort [] = []
    sort (x:xs) = sort smaller ++ [x] ++ sort larger
      where
        smaller = [y | y <- xs, y <= x]
        larger = [y | y <- xs, y > x]

main :: IO ()
main = do
    print (double 5)                    -- 10
    print (multiply 3 4)                -- 12
    print (identity "hello")            -- "hello"
    print (maximum' [1,5,3,9,2])        -- 9
    putStrLn (showAndSort [3,1,4,1,5])  -- "[1,1,3,4,5]"
```

### Type Inference Examples
```haskell
-- type_inference.hs

-- Type inferred as Num a => a -> a -> a
add x y = x + y

-- Type inferred as [a] -> Int
count xs = length xs

-- Type inferred as Eq a => a -> [a] -> Bool
contains x xs = x `elem` xs

-- Type inferred as (a -> b) -> [a] -> [b]
apply f xs = map f xs

-- Check types in GHCi:
-- :type add      -- Num a => a -> a -> a
-- :type count    -- Foldable t => t a -> Int
-- :type contains -- Eq a => a -> [a] -> Bool
-- :type apply    -- (a -> b) -> [a] -> [b]

main :: IO ()
main = do
    print (add 5 3)                     -- 8
    print (count [1,2,3,4])             -- 4
    print (contains 3 [1,2,3,4])        -- True
    print (apply (*2) [1,2,3,4])        -- [2,4,6,8]
```

## List Examples

### List Construction and Basic Operations
```haskell
-- list_operations.hs

-- Different ways to create lists
numbers1 :: [Int]
numbers1 = [1, 2, 3, 4, 5]

numbers2 :: [Int]
numbers2 = 1 : 2 : 3 : 4 : 5 : []

numbers3 :: [Int]
numbers3 = [1..5]

-- List operations
listExamples :: IO ()
listExamples = do
    print numbers1                      -- [1,2,3,4,5]
    print (head numbers1)               -- 1
    print (tail numbers1)               -- [2,3,4,5]
    print (last numbers1)               -- 5
    print (init numbers1)               -- [1,2,3,4]
    print (length numbers1)             -- 5
    print (null [])                     -- True
    print (null numbers1)               -- False
    print (numbers1 ++ [6, 7])          -- [1,2,3,4,5,6,7]
    print (0 : numbers1)                -- [0,1,2,3,4,5]

main = listExamples
```

### List Comprehensions
```haskell
-- list_comprehensions.hs

-- Basic list comprehensions
squares :: [Int]
squares = [x^2 | x <- [1..10]]

evens :: [Int]
evens = [x | x <- [1..20], even x]

-- Multiple generators
coordinates :: [(Int, Int)]
coordinates = [(x, y) | x <- [1..3], y <- [1..3]]

-- With multiple conditions
specialNumbers :: [Int]
specialNumbers = [x | x <- [1..100], x `mod` 3 == 0, x `mod` 5 == 0]

-- Pythagorean triples
pythagorean :: Int -> [(Int, Int, Int)]
pythagorean n = [(a, b, c) | c <- [1..n], 
                            b <- [1..c], 
                            a <- [1..b], 
                            a^2 + b^2 == c^2]

-- String processing
removeVowels :: String -> String
removeVowels s = [c | c <- s, not (c `elem` "aeiouAEIOU")]

main :: IO ()
main = do
    print (take 5 squares)              -- [1,4,9,16,25]
    print (take 5 evens)                -- [2,4,6,8,10]
    print coordinates                   -- [(1,1),(1,2),...,(3,3)]
    print specialNumbers                -- [15,30,45,60,75,90]
    print (pythagorean 15)              -- [(3,4,5),(6,8,10),(9,12,15)]
    putStrLn (removeVowels "Hello World") -- "Hll Wrld"
```

### Range Examples
```haskell
-- ranges.hs

-- Simple ranges
simpleRanges :: IO ()
simpleRanges = do
    print [1..10]                       -- [1,2,3,4,5,6,7,8,9,10]
    print ['a'..'z']                    -- "abcdefghijklmnopqrstuvwxyz"
    print [2, 4..20]                    -- [2,4,6,8,10,12,14,16,18,20]
    print [10, 9..1]                    -- [10,9,8,7,6,5,4,3,2,1]

-- Infinite lists (be careful!)
infiniteExamples :: IO ()
infiniteExamples = do
    print (take 10 [1..])               -- [1,2,3,4,5,6,7,8,9,10]
    print (take 5 [2, 4..])             -- [2,4,6,8,10]
    print (take 10 (cycle [1,2,3]))     -- [1,2,3,1,2,3,1,2,3,1]
    print (take 10 (repeat 5))          -- [5,5,5,5,5,5,5,5,5,5]

main :: IO ()
main = do
    simpleRanges
    infiniteExamples
```

## Tuple Examples

### Tuple Operations
```haskell
-- tuples.hs

-- Different tuple types
point :: (Int, Int)
point = (3, 4)

person :: (String, Int, Bool)
person = ("Alice", 30, True)

mixed :: (String, [Int], Char, Bool)
mixed = ("hello", [1,2,3], 'x', False)

-- Tuple functions
tupleOperations :: IO ()
tupleOperations = do
    print point                         -- (3,4)
    print (fst point)                   -- 3
    print (snd point)                   -- 4
    print person                        -- ("Alice",30,True)

-- Pattern matching with tuples
getName :: (String, Int, Bool) -> String
getName (name, _, _) = name

getAge :: (String, Int, Bool) -> Int
getAge (_, age, _) = age

isActive :: (String, Int, Bool) -> Bool
isActive (_, _, active) = active

-- Swapping tuple elements
swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)

-- Distance between two points
distance :: (Double, Double) -> (Double, Double) -> Double
distance (x1, y1) (x2, y2) = sqrt ((x2-x1)^2 + (y2-y1)^2)

main :: IO ()
main = do
    tupleOperations
    print (getName person)              -- "Alice"
    print (getAge person)               -- 30
    print (isActive person)             -- True
    print (swap (1, 2))                 -- (2,1)
    print (distance (0,0) (3,4))        -- 5.0
```

## Function Definition Examples

### Basic Functions
```haskell
-- functions.hs

-- Simple functions
square :: Int -> Int
square x = x * x

cube :: Int -> Int
cube x = x * x * x

-- Multiple parameter functions
add :: Int -> Int -> Int
add x y = x + y

subtract' :: Int -> Int -> Int
subtract' x y = x - y

-- Function using other functions
sumOfSquares :: Int -> Int -> Int
sumOfSquares x y = square x + square y

main :: IO ()
main = do
    print (square 5)                    -- 25
    print (cube 3)                      -- 27
    print (add 10 20)                   -- 30
    print (subtract' 20 5)              -- 15
    print (sumOfSquares 3 4)            -- 25
```

### Currying and Partial Application
```haskell
-- currying.hs

-- Original function
multiply :: Int -> Int -> Int
multiply x y = x * y

-- Partial applications
double :: Int -> Int
double = multiply 2

triple :: Int -> Int
triple = multiply 3

multiplyByTen :: Int -> Int
multiplyByTen = multiply 10

-- Using partial application with operators
addFive :: Int -> Int
addFive = (+5)

subtractFromTen :: Int -> Int
subtractFromTen = (10-)

divideByTwo :: Double -> Double
divideByTwo = (/2)

main :: IO ()
main = do
    print (double 7)                    -- 14
    print (triple 5)                    -- 15
    print (multiplyByTen 3)             -- 30
    print (addFive 10)                  -- 15
    print (subtractFromTen 3)           -- 7
    print (divideByTwo 20)              -- 10.0
```

### Lambda Functions
```haskell
-- lambdas.hs

-- Lambda examples
squares :: [Int] -> [Int]
squares xs = map (\x -> x * x) xs

-- Multiple parameter lambda
addLambda :: Int -> Int -> Int
addLambda = \x y -> x + y

-- Lambda with pattern matching
processEither :: [Either String Int] -> [String]
processEither xs = map (\case
    Left err -> "Error: " ++ err
    Right n -> "Success: " ++ show n) xs

-- Using lambdas with higher-order functions
processNumbers :: [Int] -> [Int]
processNumbers = filter (\x -> x > 0) . map (\x -> x * 2)

main :: IO ()
main = do
    print (squares [1,2,3,4])           -- [1,4,9,16]
    print (addLambda 5 3)               -- 8
    print (processNumbers [-2,1,-3,4])  -- [2,8]
```

## Pattern Matching Examples

### Basic Pattern Matching
```haskell
-- pattern_matching.hs

-- Pattern matching on numbers
describe :: Int -> String
describe 0 = "zero"
describe 1 = "one"
describe 2 = "two"
describe n = "some other number: " ++ show n

-- Pattern matching on booleans
boolToInt :: Bool -> Int
boolToInt True = 1
boolToInt False = 0

-- Pattern matching on characters
isVowel :: Char -> Bool
isVowel 'a' = True
isVowel 'e' = True
isVowel 'i' = True
isVowel 'o' = True
isVowel 'u' = True
isVowel _ = False

main :: IO ()
main = do
    putStrLn (describe 0)               -- "zero"
    putStrLn (describe 5)               -- "some other number: 5"
    print (boolToInt True)              -- 1
    print (isVowel 'a')                 -- True
    print (isVowel 'b')                 -- False
```

### List Pattern Matching
```haskell
-- list_patterns.hs

-- Pattern matching on lists
listLength :: [a] -> Int
listLength [] = 0
listLength (_:xs) = 1 + listLength xs

safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x

safeTail :: [a] -> Maybe [a]
safeTail [] = Nothing
safeTail (_:xs) = Just xs

-- Pattern matching with multiple elements
firstTwo :: [a] -> Maybe (a, a)
firstTwo (x:y:_) = Just (x, y)
firstTwo _ = Nothing

-- Pattern matching with @-patterns
duplicate :: [a] -> [a]
duplicate [] = []
duplicate all@(x:xs) = x : all

main :: IO ()
main = do
    print (listLength [1,2,3,4])        -- 4
    print (safeHead [1,2,3])            -- Just 1
    print (safeHead ([] :: [Int]))      -- Nothing
    print (firstTwo [1,2,3])            -- Just (1,2)
    print (firstTwo [1])                -- Nothing
    print (duplicate [1,2,3])           -- [1,1,2,3]
```

### Guards and Where Clauses
```haskell
-- guards_where.hs

-- Using guards
absoluteValue :: Int -> Int
absoluteValue x
    | x >= 0 = x
    | otherwise = -x

-- Grade calculation with guards
calculateGrade :: Int -> Char
calculateGrade score
    | score >= 90 = 'A'
    | score >= 80 = 'B'
    | score >= 70 = 'C'
    | score >= 60 = 'D'
    | otherwise = 'F'

-- Using where clause
bmiCategory :: Double -> Double -> String
bmiCategory weight height
    | bmi < 18.5 = "Underweight"
    | bmi < 25.0 = "Normal weight"
    | bmi < 30.0 = "Overweight"
    | otherwise = "Obese"
    where bmi = weight / height^2

-- Complex calculation with where
quadraticRoots :: Double -> Double -> Double -> (Double, Double)
quadraticRoots a b c = (root1, root2)
    where
        discriminant = b^2 - 4*a*c
        sqrtDisc = sqrt discriminant
        root1 = (-b + sqrtDisc) / (2*a)
        root2 = (-b - sqrtDisc) / (2*a)

main :: IO ()
main = do
    print (absoluteValue (-5))          -- 5
    print (calculateGrade 85)           -- 'B'
    putStrLn (bmiCategory 70 1.75)      -- "Normal weight"
    print (quadraticRoots 1 (-5) 6)     -- (3.0,2.0)
```

## Error Handling Examples

### Maybe Type Examples
```haskell
-- maybe_examples.hs

-- Safe division
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

-- Safe list operations
safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x

safeLast :: [a] -> Maybe a
safeLast [] = Nothing
safeLast [x] = Just x
safeLast (_:xs) = safeLast xs

-- Safe indexing
safeIndex :: [a] -> Int -> Maybe a
safeIndex [] _ = Nothing
safeIndex (x:_) 0 = Just x
safeIndex (_:xs) n
    | n < 0 = Nothing
    | otherwise = safeIndex xs (n-1)

-- Working with Maybe values
processMaybe :: Maybe Int -> String
processMaybe Nothing = "No value"
processMaybe (Just x) = "Value: " ++ show x

main :: IO ()
main = do
    print (safeDivide 10 2)             -- Just 5.0
    print (safeDivide 10 0)             -- Nothing
    print (safeHead [1,2,3])            -- Just 1
    print (safeHead ([] :: [Int]))      -- Nothing
    print (safeIndex [1,2,3,4] 2)       -- Just 3
    print (safeIndex [1,2,3,4] 5)       -- Nothing
    putStrLn (processMaybe (Just 42))   -- "Value: 42"
    putStrLn (processMaybe Nothing)     -- "No value"
```

### Either Type Examples
```haskell
-- either_examples.hs

-- Parse integer with error message
parseInt :: String -> Either String Int
parseInt s = case reads s of
    [(n, "")] -> Right n
    _ -> Left ("Could not parse: " ++ s)

-- Safe division with error messages
safeDivideEither :: Double -> Double -> Either String Double
safeDivideEither _ 0 = Left "Division by zero"
safeDivideEither x y = Right (x / y)

-- Combining Either operations
calculate :: String -> String -> Either String Double
calculate xs ys = do
    x <- parseInt xs
    y <- parseInt ys
    safeDivideEither (fromIntegral x) (fromIntegral y)

-- Processing Either results
showResult :: Either String Double -> String
showResult (Left err) = "Error: " ++ err
showResult (Right val) = "Result: " ++ show val

main :: IO ()
main = do
    print (parseInt "123")              -- Right 123
    print (parseInt "abc")              -- Left "Could not parse: abc"
    print (safeDivideEither 10 2)       -- Right 5.0
    print (safeDivideEither 10 0)       -- Left "Division by zero"
    putStrLn (showResult (calculate "10" "2"))  -- "Result: 5.0"
    putStrLn (showResult (calculate "10" "0"))  -- "Error: Division by zero"
```

## Running Examples

To run any of these examples:

1. Save to a `.hs` file
2. Compile: `ghc filename.hs`
3. Run: `./filename`

Or use GHCi:
```bash
ghci filename.hs
*Main> main
```

For testing individual functions:
```bash
ghci filename.hs
*Main> :type functionName
*Main> functionName arguments
```