# Functions and Higher-Order Functions - Notes

## Overview

Functions are the fundamental building blocks of Haskell. Understanding functions deeply—from basic definition to advanced higher-order patterns—is crucial for effective Haskell programming. This section explores function composition, currying, partial application, and the powerful higher-order functions that make functional programming so expressive.

## Function Fundamentals

### Function Definition Syntax

```haskell
-- Basic function definition
functionName :: Type1 -> Type2 -> ... -> ReturnType
functionName parameter1 parameter2 = expression

-- Example
add :: Int -> Int -> Int
add x y = x + y
```

### Function Application

```haskell
-- Function application (left-associative)
f x y z = ((f x) y) z

-- Examples
add 3 4          -- 7
multiply 2 5     -- 10
```

### Function Types

```haskell
-- Function type constructor (->)
(+) :: Num a => a -> a -> a
map :: (a -> b) -> [a] -> [b]
filter :: (a -> Bool) -> [a] -> [a]
```

## Currying and Partial Application

### Understanding Currying

All functions in Haskell are automatically curried - they take one argument and return a function that takes the next argument.

```haskell
-- These are equivalent
add :: Int -> Int -> Int
add :: Int -> (Int -> Int)

-- Partial application
addFive :: Int -> Int
addFive = add 5

-- Using partial application
increment = (+1)
double = (*2)
halve = (/2)
```

### Benefits of Currying

1. **Partial Application**: Create specialized functions
2. **Function Composition**: Easier to combine functions
3. **Code Reuse**: Build complex functions from simple ones
4. **Point-Free Style**: Write functions without explicitly mentioning arguments

### Examples of Partial Application

```haskell
-- Creating specialized functions
multiplyBy :: Num a => a -> (a -> a)
multiplyBy n = (*n)

double = multiplyBy 2
triple = multiplyBy 3

-- Operator sections
greaterThanFive = (>5)
addToTen = (10+)
subtractFromTen = (10-)
```

## Function Composition

### The Composition Operator (.)

```haskell
-- Function composition
(.) :: (b -> c) -> (a -> b) -> (a -> c)
(f . g) x = f (g x)

-- Examples
doubleAndSquare :: Int -> Int
doubleAndSquare = square . double
  where
    square x = x * x
    double x = x * 2

-- Multiple composition
processText :: String -> String
processText = reverse . map toUpper . words . filter isAlpha
```

### Benefits of Composition

1. **Readability**: Express data transformations clearly
2. **Modularity**: Build complex operations from simple parts
3. **Reusability**: Compose existing functions in new ways
4. **Reasoning**: Easier to understand and verify correctness

### Application Operator ($)

```haskell
-- Application operator (function application with lowest precedence)
($) :: (a -> b) -> a -> b
f $ x = f x

-- Useful for avoiding parentheses
sqrt $ 3 + 4 * 5    -- Instead of sqrt (3 + 4 * 5)
```

## Higher-Order Functions

### Definition

Higher-order functions are functions that:
1. Take other functions as arguments, OR
2. Return functions as results, OR
3. Both

### Essential Higher-Order Functions

#### Map
```haskell
map :: (a -> b) -> [a] -> [b]

-- Examples
map (*2) [1,2,3,4]           -- [2,4,6,8]
map length ["hello", "world"] -- [5,5]
map (map (*2)) [[1,2],[3,4]] -- [[2,4],[6,8]]
```

#### Filter
```haskell
filter :: (a -> Bool) -> [a] -> [a]

-- Examples
filter even [1,2,3,4,5,6]    -- [2,4,6]
filter (>5) [1,8,3,9,2,7]    -- [8,9,7]
filter null [[], [1], [], [2,3]] -- [[],[]]
```

#### Fold (Reduce)
```haskell
-- Right fold
foldr :: (a -> b -> b) -> b -> [a] -> b
foldr f z []     = z
foldr f z (x:xs) = f x (foldr f z xs)

-- Left fold
foldl :: (b -> a -> b) -> b -> [a] -> b
foldl f z []     = z
foldl f z (x:xs) = foldl f (f z x) xs

-- Examples
foldr (+) 0 [1,2,3,4]        -- 10
foldl (*) 1 [1,2,3,4]        -- 24
foldr (:) [] [1,2,3]         -- [1,2,3]
```

#### Strict Left Fold
```haskell
-- Strict left fold (prevents space leaks)
foldl' :: (b -> a -> b) -> b -> [a] -> b

-- Use foldl' for performance in most cases
sumList = foldl' (+) 0
productList = foldl' (*) 1
```

### Advanced Higher-Order Patterns

#### Zip and ZipWith
```haskell
zip :: [a] -> [b] -> [(a,b)]
zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]

-- Examples
zip [1,2,3] ['a','b','c']        -- [(1,'a'),(2,'b'),(3,'c')]
zipWith (+) [1,2,3] [4,5,6]      -- [5,7,9]
zipWith (*) [1,2,3] [4,5,6]      -- [4,10,18]
```

#### TakeWhile and DropWhile
```haskell
takeWhile :: (a -> Bool) -> [a] -> [a]
dropWhile :: (a -> Bool) -> [a] -> [a]

-- Examples
takeWhile (<5) [1,2,3,6,4,7]     -- [1,2,3]
dropWhile (<5) [1,2,3,6,4,7]     -- [6,4,7]
```

#### All and Any
```haskell
all :: (a -> Bool) -> [a] -> Bool
any :: (a -> Bool) -> [a] -> Bool

-- Examples
all even [2,4,6,8]              -- True
any odd [2,4,6,8]               -- False
```

## Function Patterns and Idioms

### Point-Free Style

Writing functions without explicitly mentioning their arguments:

```haskell
-- Point-free style
sumOfSquares :: [Int] -> Int
sumOfSquares = sum . map (^2)

-- Equivalent point-full style
sumOfSquares' :: [Int] -> Int
sumOfSquares' xs = sum (map (^2) xs)

-- More examples
isEven :: Int -> Bool
isEven = (==0) . (`mod` 2)

countWords :: String -> Int
countWords = length . words
```

### Function Pipelines

```haskell
-- Using function composition for data pipelines
processData :: [String] -> [Int]
processData = map length . filter (not . null) . map (filter isAlpha)

-- Alternative with application operator
processData' :: [String] -> [Int]
processData' xs = map length $ filter (not . null) $ map (filter isAlpha) xs
```

### Combinators

Functions that combine other functions:

```haskell
-- Identity combinator
identity :: a -> a
identity x = x

-- Constant combinator
const :: a -> b -> a
const x _ = x

-- Flip combinator
flip :: (a -> b -> c) -> b -> a -> c
flip f x y = f y x

-- Examples
map (const 0) [1,2,3,4]         -- [0,0,0,0]
zipWith (flip div) [1,2,3] [10,20,30] -- [10,10,10]
```

## Lambda Functions

### Lambda Syntax

```haskell
-- Lambda function syntax
\parameter1 parameter2 -> expression

-- Examples
(\x -> x * 2)
(\x y -> x + y)
(\(x,y) -> x * y)
```

### When to Use Lambda Functions

1. **Short, one-off functions**
2. **Inline function arguments**
3. **Quick transformations**
4. **Pattern matching in anonymous functions**

```haskell
-- Lambda examples
squares = map (\x -> x * x) [1,2,3,4]
evenNumbers = filter (\x -> x `mod` 2 == 0) [1..10]
```

### Lambda with Pattern Matching

```haskell
-- Using LambdaCase extension
{-# LANGUAGE LambdaCase #-}

processEither :: [Either String Int] -> [String]
processEither = map (\case
    Left err -> "Error: " ++ err
    Right n -> "Success: " ++ show n)
```

## Advanced Function Concepts

### Recursive Functions as Higher-Order

```haskell
-- Higher-order recursive function
mapTree :: (a -> b) -> Tree a -> Tree b
mapTree f Empty = Empty
mapTree f (Node x left right) = Node (f x) (mapTree f left) (mapTree f right)

-- Fold for trees
foldTree :: (a -> b -> b -> b) -> b -> Tree a -> b
foldTree f z Empty = z
foldTree f z (Node x left right) = f x (foldTree f z left) (foldTree f z right)
```

### Function Factories

Functions that create other functions:

```haskell
-- Creating validators
makeValidator :: (a -> Bool) -> String -> (a -> Either String a)
makeValidator predicate errorMsg = \x ->
    if predicate x
        then Right x
        else Left errorMsg

-- Usage
positiveValidator = makeValidator (>0) "Must be positive"
nonEmptyValidator = makeValidator (not . null) "Must not be empty"
```

### Memoization

Caching function results for performance:

```haskell
-- Simple memoization for fibonacci
memoFib :: Int -> Integer
memoFib = (map fib [0..] !!)
  where
    fib 0 = 0
    fib 1 = 1
    fib n = memoFib (n-1) + memoFib (n-2)
```

## Research Papers

### Foundational Papers

1. **"Can Programming Be Liberated from the von Neumann Style?" (1978)**
   - Author: John Backus
   - [Link](https://dl.acm.org/doi/10.1145/359576.359579)
   - Introduces functional programming concepts and higher-order functions

2. **"Why Functional Programming Matters" (1990)**
   - Author: John Hughes
   - [Link](https://www.cs.kent.ac.uk/people/staff/dat/miranda/whyfp90.pdf)
   - Demonstrates the power of higher-order functions and lazy evaluation

3. **"Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire" (1991)**
   - Authors: Erik Meijer, Maarten Fokkinga, Ross Paterson
   - [Link](https://research.microsoft.com/en-us/um/people/emeijer/papers/fpca91.pdf)
   - Introduces recursion schemes and generic programming

### Advanced Topics

1. **"Generalising Monads to Arrows" (2000)**
   - Author: John Hughes
   - Extends function composition to more general computation patterns

2. **"The Essence of the Iterator Pattern" (2006)**
   - Authors: Jeremy Gibbons and Bruno Oliveira
   - Demonstrates applicative functors through higher-order functions

## Performance Considerations

### Efficiency of Higher-Order Functions

1. **Fusion**: GHC can optimize chains of higher-order functions
2. **Inlining**: Small functions get inlined for better performance
3. **Strictness**: Use strict folds when building up large data structures

```haskell
-- Good: Fusion optimization possible
processNumbers :: [Int] -> [Int]
processNumbers = map (*2) . filter (>0) . map (+1)

-- Less optimal: Intermediate lists created
processNumbers' :: [Int] -> [Int]
processNumbers' xs = 
    let step1 = map (+1) xs
        step2 = filter (>0) step1
        step3 = map (*2) step2
    in step3
```

### Space Considerations

```haskell
-- Space leak with foldl
badSum = foldl (+) 0  -- Builds up thunks

-- Good: Strict evaluation
goodSum = foldl' (+) 0  -- Evaluates immediately
```

## Best Practices

### Function Design

1. **Small, focused functions**: Each function should do one thing well
2. **Pure functions**: Avoid side effects when possible
3. **Meaningful names**: Function names should describe their purpose
4. **Type signatures**: Always provide type signatures for top-level functions

### Composition Guidelines

1. **Read left to right**: Use (.) for data transformation pipelines
2. **Use ($) judiciously**: Avoid parentheses where it improves readability
3. **Point-free when clear**: Use point-free style when it enhances clarity
4. **Partial application**: Leverage currying for reusable functions

### Common Patterns

```haskell
-- Pattern: Processing pipelines
analyzeText :: String -> (Int, Int, [String])
analyzeText text = 
    let ws = words text
        wordCount = length ws
        charCount = length . filter (not . isSpace) $ text
        longWords = filter ((>5) . length) ws
    in (wordCount, charCount, longWords)

-- Pattern: Validation chains
validateUser :: User -> Either String User
validateUser user = 
    validateEmail (email user) >>
    validateAge (age user) >>
    validateName (name user) >>
    return user
```

## Common Mistakes and How to Avoid Them

### Mistake 1: Overusing Point-Free Style

```haskell
-- Too obscure
mysterious = (.) . (.) . (.)

-- Better: Clear and readable
compose3 f g h x = f (g (h x))
```

### Mistake 2: Ignoring Laziness

```haskell
-- May cause space leaks
processLargeList = foldl (+) 0 . map expensiveFunction

-- Better: Use strict fold
processLargeList' = foldl' (+) 0 . map expensiveFunction
```

### Mistake 3: Not Leveraging Partial Application

```haskell
-- Verbose
isPositive x = x > 0
doubleList xs = map (\x -> x * 2) xs

-- More idiomatic
isPositive = (>0)
doubleList = map (*2)
```

This comprehensive understanding of functions and higher-order patterns forms the foundation for advanced Haskell programming and leads naturally into more sophisticated abstractions like functors and monads.