# Comprehensive Guide to Haskell Programming

## Table of Contents

1. [Introduction to Haskell](#1-introduction-to-haskell)
2. [Basic Syntax and Types](#2-basic-syntax-and-types)
3. [Functions and Higher-Order Functions](#3-functions-and-higher-order-functions)
4. [Pattern Matching and Recursion](#4-pattern-matching-and-recursion)
5. [Algebraic Data Types](#5-algebraic-data-types)
6. [Type Classes and Polymorphism](#6-type-classes-and-polymorphism)
7. [Lazy Evaluation](#7-lazy-evaluation)
8. [Monads and Functors](#8-monads-and-functors)
9. [Advanced Type System Features](#9-advanced-type-system-features)
10. [Concurrency and Parallelism](#10-concurrency-and-parallelism)
11. [Performance and Optimization](#11-performance-and-optimization)
12. [Real-World Applications](#12-real-world-applications)

---

## 1. Introduction to Haskell

### Overview
Haskell is a statically typed, purely functional programming language named after logician Haskell Curry. It emphasizes immutability, lazy evaluation, and mathematical elegance.

### Key Features
- **Pure functions**: No side effects
- **Lazy evaluation**: Expressions evaluated only when needed
- **Strong static typing**: Type safety at compile time
- **Higher-order functions**: Functions as first-class values
- **Pattern matching**: Elegant data deconstruction

### Hello World Example
```haskell
-- Simple Hello World
main :: IO ()
main = putStrLn "Hello, World!"

-- Interactive version
main :: IO ()
main = do
    putStrLn "What's your name?"
    name <- getLine
    putStrLn ("Hello, " ++ name ++ "!")
```

### Research Papers
- **"A History of Haskell: Being Lazy with Class" (2007)** - Paul Hudak et al.
  - [Link](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/history.pdf)
- **"Haskell 2010 Language Report"** - Simon Marlow (editor)
  - [Link](https://www.haskell.org/definition/haskell2010.pdf)

### Simple Project: Number Guessing Game
```haskell
import System.Random

main :: IO ()
main = do
    secret <- randomRIO (1, 100) :: IO Int
    putStrLn "Guess a number between 1 and 100:"
    guessLoop secret

guessLoop :: Int -> IO ()
guessLoop secret = do
    input <- getLine
    let guess = read input :: Int
    case compare guess secret of
        LT -> do putStrLn "Too low!"; guessLoop secret
        GT -> do putStrLn "Too high!"; guessLoop secret
        EQ -> putStrLn "Correct!"
```

### Recommended Reading
- *Learn You a Haskell for Great Good!* by Miran Lipovača
- *Programming in Haskell* by Graham Hutton
- *Real World Haskell* by Bryan O'Sullivan, Don Stewart, and John Goerzen

---

## 2. Basic Syntax and Types

### Primitive Types
```haskell
-- Numbers
intValue :: Int
intValue = 42

floatValue :: Float
floatValue = 3.14159

doubleValue :: Double
doubleValue = 2.718281828

-- Booleans
isTrue :: Bool
isTrue = True

isFalse :: Bool
isFalse = False

-- Characters and Strings
singleChar :: Char
singleChar = 'A'

greeting :: String  -- String is [Char]
greeting = "Hello, Haskell!"
```

### Lists and Tuples
```haskell
-- Lists (homogeneous)
numbers :: [Int]
numbers = [1, 2, 3, 4, 5]

-- List operations
firstElement = head numbers      -- 1
restElements = tail numbers      -- [2,3,4,5]
listLength = length numbers      -- 5

-- Tuples (can be heterogeneous)
person :: (String, Int, Bool)
person = ("Alice", 30, True)

-- Accessing tuple elements
getName :: (String, Int, Bool) -> String
getName (name, _, _) = name
```

### Type Signatures and Inference
```haskell
-- Explicit type signatures
add :: Int -> Int -> Int
add x y = x + y

-- Type inference (compiler figures out the type)
multiply x y = x * y  -- Inferred as Num a => a -> a -> a

-- Polymorphic functions
identity :: a -> a
identity x = x
```

### Research Papers
- **"Principal Type-Schemes for Functional Programs" (1982)** - Luis Damas and Robin Milner
  - [Link](https://web.cs.wpi.edu/~cs4536/c12/milner-damas_principal_types.pdf)
- **"Type Classes in Haskell" (1996)** - Cordelia Hall et al.
  - [Link](https://dl.acm.org/doi/10.1145/227699.227700)

### Simple Project: Calculator
```haskell
data Operation = Add | Subtract | Multiply | Divide
    deriving (Show, Eq)

calculate :: Operation -> Double -> Double -> Double
calculate Add x y = x + y
calculate Subtract x y = x - y
calculate Multiply x y = x * y
calculate Divide x y = if y /= 0 then x / y else error "Division by zero"

main :: IO ()
main = do
    putStrLn "Enter first number:"
    num1 <- readLn
    putStrLn "Enter operation (+, -, *, /):"
    op <- getLine
    putStrLn "Enter second number:"
    num2 <- readLn
    
    let operation = case op of
            "+" -> Add
            "-" -> Subtract
            "*" -> Multiply
            "/" -> Divide
            _ -> error "Invalid operation"
    
    let result = calculate operation num1 num2
    putStrLn $ "Result: " ++ show result
```

---

## 3. Functions and Higher-Order Functions

### Function Definition and Application
```haskell
-- Basic function definition
square :: Int -> Int
square x = x * x

-- Function with multiple parameters
power :: Int -> Int -> Int
power base exponent = base ^ exponent

-- Currying and partial application
addFive :: Int -> Int
addFive = (+) 5  -- Partially applied function

-- Function composition
compose :: (b -> c) -> (a -> b) -> (a -> c)
compose f g x = f (g x)

-- Using composition operator (.)
doubleAndSquare :: Int -> Int
doubleAndSquare = square . (*2)
```

### Higher-Order Functions
```haskell
-- map: Apply function to each element
doubleList :: [Int] -> [Int]
doubleList = map (*2)

-- filter: Keep elements that satisfy predicate
evens :: [Int] -> [Int]
evens = filter even

-- fold: Reduce list to single value
sumList :: [Int] -> Int
sumList = foldr (+) 0

-- Custom higher-order function
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)

-- Function that returns a function
makeAdder :: Int -> (Int -> Int)
makeAdder n = \x -> x + n
```

### Lambda Functions
```haskell
-- Anonymous functions
squareList :: [Int] -> [Int]
squareList xs = map (\x -> x * x) xs

-- Multi-parameter lambda
addLambda :: Int -> Int -> Int
addLambda = \x y -> x + y

-- Lambda with pattern matching
processEither :: [Either String Int] -> [Int]
processEither = map (\case
    Left _ -> 0
    Right n -> n)
```

### Research Papers
- **"Can Programming Be Liberated from the von Neumann Style?" (1978)** - John Backus
  - [Link](https://dl.acm.org/doi/10.1145/359576.359579)
- **"Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire" (1991)** - Erik Meijer et al.
  - [Link](https://research.microsoft.com/en-us/um/people/emeijer/papers/fpca91.pdf)

### Simple Project: Text Processing Tool
```haskell
import Data.Char (toLower, isAlpha)
import Data.List (sort, group)

-- Word frequency counter
wordFrequency :: String -> [(String, Int)]
wordFrequency text = 
    map (\ws -> (head ws, length ws)) $
    group $ sort $
    filter (not . null) $
    map (filter isAlpha . map toLower) $
    words text

-- Text statistics
data TextStats = TextStats
    { totalWords :: Int
    , totalChars :: Int
    , averageWordLength :: Double
    , longestWord :: String
    } deriving Show

analyzeText :: String -> TextStats
analyzeText text = 
    let ws = words text
        lengths = map length ws
        total = length ws
        chars = length $ filter isAlpha text
        avgLen = if total > 0 then fromIntegral chars / fromIntegral total else 0
        longest = if null ws then "" else maximumBy (\a b -> compare (length a) (length b)) ws
    in TextStats total chars avgLen longest

main :: IO ()
main = do
    putStrLn "Enter text to analyze:"
    text <- getLine
    let stats = analyzeText text
    let freq = take 5 $ wordFrequency text
    
    putStrLn $ "Statistics: " ++ show stats
    putStrLn $ "Top 5 words: " ++ show freq
```

---

## 4. Pattern Matching and Recursion

### Pattern Matching Fundamentals
```haskell
-- Basic pattern matching
describe :: Int -> String
describe 0 = "zero"
describe 1 = "one"
describe 2 = "two"
describe n = "many: " ++ show n

-- Pattern matching on lists
listLength :: [a] -> Int
listLength [] = 0
listLength (_:xs) = 1 + listLength xs

-- Pattern matching on tuples
addPair :: (Int, Int) -> Int
addPair (x, y) = x + y

-- Guards
absoluteValue :: Int -> Int
absoluteValue n
    | n >= 0 = n
    | otherwise = -n
```

### Advanced Pattern Matching
```haskell
-- Nested patterns
processNestedList :: [[Int]] -> Int
processNestedList [] = 0
processNestedList ([]:xss) = processNestedList xss
processNestedList ((y:ys):xss) = y + processNestedList (ys:xss)

-- As-patterns
duplicate :: [a] -> [a]
duplicate [] = []
duplicate all@(x:xs) = x : all

-- Pattern matching with where clauses
bmiCategory :: Float -> Float -> String
bmiCategory weight height
    | bmi <= 18.5 = "underweight"
    | bmi <= 25.0 = "normal"
    | bmi <= 30.0 = "overweight"
    | otherwise = "obese"
    where bmi = weight / height^2
```

### Recursion Patterns
```haskell
-- Linear recursion
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Tail recursion
factorialTail :: Int -> Int
factorialTail n = factorialHelper n 1
  where
    factorialHelper 0 acc = acc
    factorialHelper n acc = factorialHelper (n - 1) (n * acc)

-- Tree recursion
fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

-- Mutual recursion
even' :: Int -> Bool
even' 0 = True
even' n = odd' (n - 1)

odd' :: Int -> Bool
odd' 0 = False
odd' n = even' (n - 1)
```

### Research Papers
- **"Recursion Schemes for Dynamic Programming" (2004)** - Jeremy Gibbons
  - [Link](https://www.cs.ox.ac.uk/jeremy.gibbons/publications/dynprog.pdf)
- **"The Essence of Functional Programming" (1992)** - Philip Wadler
  - [Link](https://jgbm.github.io/eecs762f19/papers/wadler-essence.pdf)

### Simple Project: Binary Tree Operations
```haskell
-- Binary tree definition
data Tree a = Empty | Node a (Tree a) (Tree a)
    deriving (Show, Eq)

-- Tree operations using recursion and pattern matching
insert :: Ord a => a -> Tree a -> Tree a
insert x Empty = Node x Empty Empty
insert x (Node y left right)
    | x <= y = Node y (insert x left) right
    | otherwise = Node y left (insert x right)

search :: Ord a => a -> Tree a -> Bool
search _ Empty = False
search x (Node y left right)
    | x == y = True
    | x < y = search x left
    | otherwise = search x right

inorder :: Tree a -> [a]
inorder Empty = []
inorder (Node x left right) = inorder left ++ [x] ++ inorder right

treeHeight :: Tree a -> Int
treeHeight Empty = 0
treeHeight (Node _ left right) = 1 + max (treeHeight left) (treeHeight right)

-- Example usage
exampleTree :: Tree Int
exampleTree = foldr insert Empty [5, 3, 7, 1, 9, 4, 6]

main :: IO ()
main = do
    print exampleTree
    print $ search 4 exampleTree
    print $ inorder exampleTree
    print $ treeHeight exampleTree
```

---

## 5. Algebraic Data Types

### Sum Types (Union Types)
```haskell
-- Basic sum type
data Color = Red | Green | Blue
    deriving (Show, Eq)

-- Sum type with parameters
data Shape = Circle Float | Rectangle Float Float | Triangle Float Float Float
    deriving Show

area :: Shape -> Float
area (Circle r) = pi * r * r
area (Rectangle w h) = w * h
area (Triangle a b c) = 
    let s = (a + b + c) / 2
    in sqrt (s * (s - a) * (s - b) * (s - c))

-- Maybe type (built-in)
safeDivide :: Float -> Float -> Maybe Float
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

-- Either type for error handling
data ParseError = EmptyString | InvalidNumber String
    deriving Show

parseNumber :: String -> Either ParseError Int
parseNumber "" = Left EmptyString
parseNumber s = case reads s of
    [(n, "")] -> Right n
    _ -> Left (InvalidNumber s)
```

### Product Types and Records
```haskell
-- Product type
data Point = Point Float Float
    deriving Show

distance :: Point -> Point -> Float
distance (Point x1 y1) (Point x2 y2) = 
    sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Record syntax
data Person = Person
    { name :: String
    , age :: Int
    , email :: String
    } deriving Show

-- Creating and updating records
john :: Person
john = Person "John Doe" 30 "john@example.com"

johnOlder :: Person
johnOlder = john { age = 31 }

-- Accessing record fields
getPersonInfo :: Person -> String
getPersonInfo p = name p ++ " is " ++ show (age p) ++ " years old"
```

### Recursive Data Types
```haskell
-- List reimplementation
data List a = Nil | Cons a (List a)
    deriving Show

listMap :: (a -> b) -> List a -> List b
listMap _ Nil = Nil
listMap f (Cons x xs) = Cons (f x) (listMap f xs)

-- Expression tree
data Expr = Num Int
          | Add Expr Expr
          | Mul Expr Expr
          | Var String
    deriving Show

eval :: Expr -> [(String, Int)] -> Int
eval (Num n) _ = n
eval (Add e1 e2) env = eval e1 env + eval e2 env
eval (Mul e1 e2) env = eval e1 env * eval e2 env
eval (Var x) env = case lookup x env of
    Just n -> n
    Nothing -> error $ "Variable " ++ x ++ " not found"
```

### Research Papers
- **"Algebraic Data Types in Haskell" (1991)** - Philip Wadler
- **"Views: A Way for Pattern Matching to Cohabit with Data Abstraction" (1987)** - Philip Wadler
  - [Link](https://www.microsoft.com/en-us/research/wp-content/uploads/1987/01/views.pdf)

### Simple Project: JSON Parser
```haskell
import Data.Char (isDigit, isSpace)

-- JSON value representation
data JSON = JNull
          | JBool Bool
          | JNumber Double
          | JString String
          | JArray [JSON]
          | JObject [(String, JSON)]
    deriving (Show, Eq)

-- Simple JSON parser (simplified)
parseJSON :: String -> Maybe JSON
parseJSON s = case parseValue (dropWhile isSpace s) of
    Just (value, "") -> Just value
    _ -> Nothing

parseValue :: String -> Maybe (JSON, String)
parseValue ('n':'u':'l':'l':rest) = Just (JNull, rest)
parseValue ('t':'r':'u':'e':rest) = Just (JBool True, rest)
parseValue ('f':'a':'l':'s':'e':rest) = Just (JBool False, rest)
parseValue ('"':rest) = parseString rest
parseValue s@(c:_) 
    | isDigit c || c == '-' = parseNumber s
parseValue _ = Nothing

parseString :: String -> Maybe (JSON, String)
parseString s = case break (== '"') s of
    (str, '"':rest) -> Just (JString str, rest)
    _ -> Nothing

parseNumber :: String -> Maybe (JSON, String)
parseNumber s = case reads s of
    [(n, rest)] -> Just (JNumber n, rest)
    _ -> Nothing

-- Example usage
exampleJSON :: String
exampleJSON = "\"hello\""

main :: IO ()
main = print $ parseJSON exampleJSON
```

---

## 6. Type Classes and Polymorphism

### Basic Type Classes
```haskell
-- Eq type class for equality
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    x /= y = not (x == y)  -- Default implementation

-- Show type class for string representation
class Show a where
    show :: a -> String

-- Ord type class for ordering
class Eq a => Ord a where
    compare :: a -> a -> Ordering
    (<), (<=), (>), (>=) :: a -> a -> Bool
    max, min :: a -> a -> a
```

### Creating Custom Type Classes
```haskell
-- Custom type class for things that can be serialized
class Serializable a where
    serialize :: a -> String
    deserialize :: String -> Maybe a

-- Instance for Int
instance Serializable Int where
    serialize = show
    deserialize s = case reads s of
        [(n, "")] -> Just n
        _ -> Nothing

-- Instance for custom type
data Status = Active | Inactive | Pending
    deriving Show

instance Serializable Status where
    serialize Active = "active"
    serialize Inactive = "inactive"
    serialize Pending = "pending"
    
    deserialize "active" = Just Active
    deserialize "inactive" = Just Inactive
    deserialize "pending" = Just Pending
    deserialize _ = Nothing
```

### Multi-Parameter Type Classes
```haskell
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

-- Type class for conversion between types
class Convert a b where
    convert :: a -> b

instance Convert Int Float where
    convert = fromIntegral

instance Convert String Int where
    convert s = case reads s of
        [(n, "")] -> n
        _ -> 0

-- Collection type class
class Collection c where
    empty :: c a
    insert :: a -> c a -> c a
    member :: Eq a => a -> c a -> Bool

instance Collection [] where
    empty = []
    insert = (:)
    member = elem
```

### Functor, Applicative, and Monad
```haskell
-- Functor: things that can be mapped over
class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- Maybe is a Functor
instance Functor Maybe where
    fmap _ Nothing = Nothing
    fmap f (Just x) = Just (f x)

-- Custom data type as Functor
data Box a = Box a deriving Show

instance Functor Box where
    fmap f (Box x) = Box (f x)

-- Applicative: functors with application
class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

instance Applicative Maybe where
    pure = Just
    Nothing <*> _ = Nothing
    (Just f) <*> something = fmap f something

-- Monad: sequential computation
class Applicative m => Monad m where
    (>>=) :: m a -> (a -> m b) -> m b
    return :: a -> m a
    return = pure

instance Monad Maybe where
    Nothing >>= _ = Nothing
    (Just x) >>= f = f x
```

### Research Papers
- **"How to Make Ad-hoc Polymorphism Less Ad Hoc" (1989)** - Philip Wadler and Stephen Blott
  - [Link](https://people.csail.mit.edu/dnj/teaching/6898/papers/wadler88.pdf)
- **"Type Classes: An Exploration of the Design Space" (1997)** - Simon Peyton Jones et al.

### Simple Project: Generic Data Structure Library
```haskell
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

-- Generic container interface
class Container c where
    empty :: c a
    size :: c a -> Int
    insert :: a -> c a -> c a
    toList :: c a -> [a]

-- Stack implementation
newtype Stack a = Stack [a] deriving Show

instance Container Stack where
    empty = Stack []
    size (Stack xs) = length xs
    insert x (Stack xs) = Stack (x:xs)
    toList (Stack xs) = xs

-- Queue implementation
data Queue a = Queue [a] [a] deriving Show

instance Container Queue where
    empty = Queue [] []
    size (Queue front back) = length front + length back
    insert x (Queue front back) = Queue front (x:back)
    toList (Queue front back) = front ++ reverse back

-- Generic operations
peek :: Container c => c a -> Maybe a
peek container = case toList container of
    [] -> Nothing
    (x:_) -> Just x

-- Usage example
main :: IO ()
main = do
    let stack = insert 3 $ insert 2 $ insert 1 $ empty :: Stack Int
    let queue = insert 3 $ insert 2 $ insert 1 $ empty :: Queue Int
    
    putStrLn $ "Stack: " ++ show stack
    putStrLn $ "Queue: " ++ show queue
    putStrLn $ "Stack peek: " ++ show (peek stack)
    putStrLn $ "Queue peek: " ++ show (peek queue)
```

---

## 7. Lazy Evaluation

### Understanding Lazy Evaluation
```haskell
-- Infinite lists are possible due to laziness
naturals :: [Int]
naturals = [0..]

-- Taking elements from infinite list
firstTen :: [Int]
firstTen = take 10 naturals

-- Infinite list of Fibonacci numbers
fibs :: [Integer]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

-- Sieve of Eratosthenes (infinite primes)
primes :: [Int]
primes = sieve [2..]
  where
    sieve (p:xs) = p : sieve [x | x <- xs, x `mod` p /= 0]
```

### Lazy Data Structures
```haskell
-- Lazy binary tree
data LazyTree a = Empty | Node a (LazyTree a) (LazyTree a)

-- Infinite binary tree
infiniteTree :: Int -> LazyTree Int
infiniteTree n = Node n (infiniteTree (2*n)) (infiniteTree (2*n + 1))

-- Breadth-first traversal of infinite tree
bfsTraversal :: LazyTree a -> [a]
bfsTraversal tree = bfs [tree]
  where
    bfs [] = []
    bfs (Empty : ts) = bfs ts
    bfs (Node x l r : ts) = x : bfs (ts ++ [l, r])

-- Stream data type
data Stream a = Stream a (Stream a)

-- Stream operations
streamHead :: Stream a -> a
streamHead (Stream x _) = x

streamTail :: Stream a -> Stream a
streamTail (Stream _ xs) = xs

streamTake :: Int -> Stream a -> [a]
streamTake 0 _ = []
streamTake n (Stream x xs) = x : streamTake (n-1) xs

-- Infinite stream of ones
ones :: Stream Int
ones = Stream 1 ones
```

### Controlling Evaluation
```haskell
-- seq forces evaluation
forceEval :: Int -> Int -> Int
forceEval x y = x `seq` y `seq` x + y

-- ($!) strict application
strictMap :: (a -> b) -> [a] -> [b]
strictMap f [] = []
strictMap f (x:xs) = let y = f x in y `seq` y : strictMap f xs

-- deepseq for complete evaluation
import Control.DeepSeq

data Person = Person String Int deriving Show

instance NFData Person where
    rnf (Person name age) = rnf name `seq` rnf age

-- Strict accumulator pattern
sumStrict :: [Int] -> Int
sumStrict = sumStrict' 0
  where
    sumStrict' acc [] = acc
    sumStrict' acc (x:xs) = let acc' = acc + x 
                           in acc' `seq` sumStrict' acc' xs
```

### Research Papers
- **"Lazy Evaluation and the Logic Variable" (1976)** - David H. D. Warren
- **"Implementing Lazy Functional Languages on Stock Hardware" (1992)** - Simon Peyton Jones
  - [Link](https://www.microsoft.com/en-us/research/wp-content/uploads/1992/04/spineless-tagless-gmachine.pdf)

### Simple Project: Lazy Data Processing Pipeline
```haskell
import System.IO
import Control.Exception (bracket)

-- Lazy file processing
processLargeFile :: FilePath -> FilePath -> IO ()
processLargeFile input output = do
    bracket (openFile input ReadMode) hClose $ \inputHandle ->
        bracket (openFile output WriteMode) hClose $ \outputHandle -> do
            contents <- hGetContents inputHandle
            let processed = processLines (lines contents)
            mapM_ (hPutStrLn outputHandle) processed

-- Lazy line processing
processLines :: [String] -> [String]
processLines = map processLine . filter (not . null) . map (dropWhile (== ' '))

processLine :: String -> String
processLine line = "Processed: " ++ line

-- Stream-based number generation
data NumberStream = NumberStream
    { current :: Integer
    , next :: NumberStream
    }

-- Collatz sequence (lazy)
collatz :: Integer -> [Integer]
collatz 1 = [1]
collatz n
    | even n = n : collatz (n `div` 2)
    | otherwise = n : collatz (3 * n + 1)

-- Find numbers with long Collatz sequences
longCollatzNumbers :: [(Integer, Int)]
longCollatzNumbers = filter ((> 100) . snd) $ 
                    map (\n -> (n, length $ collatz n)) [1..]

main :: IO ()
main = do
    putStrLn "First 10 Fibonacci numbers:"
    print $ take 10 fibs
    
    putStrLn "\nFirst 10 primes:"
    print $ take 10 primes
    
    putStrLn "\nNumbers with Collatz sequences longer than 100:"
    print $ take 5 longCollatzNumbers
```