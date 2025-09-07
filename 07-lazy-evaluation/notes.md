# Lazy Evaluation - Comprehensive Notes

## Overview

Lazy evaluation is one of Haskell's defining features, allowing computations to be deferred until their results are actually needed. This enables powerful programming patterns like infinite data structures and efficient memory usage, but requires understanding to avoid space leaks and performance pitfalls.

**Key Learning Objectives:**
- Understand the mechanics of lazy evaluation and thunks
- Learn to leverage laziness for elegant solutions
- Master techniques to control evaluation when needed
- Identify and avoid common lazy evaluation pitfalls
- Apply lazy evaluation patterns to solve real-world problems
- Optimize Haskell programs using lazy evaluation effectively

## How Lazy Evaluation Works: The Mechanics

### Thunks: Deferred Computations

A thunk is an unevaluated computation that is stored in memory and computed only when needed.

**Understanding Thunks:**
```haskell
-- Simple thunk example
x = 1 + 2  -- This creates a thunk, not the value 3

-- Thunk evaluation
y = x + 1  -- This creates another thunk: (1 + 2) + 1

-- Force evaluation
z = x `seq` x  -- Forces evaluation of x, returns 3

-- Thunk structure in memory
-- x -> Thunk: 1 + 2
-- y -> Thunk: x + 1
-- z -> Value: 3 (after evaluation)
```

**Thunk Lifecycle:**
```haskell
-- 1. Creation: Expression stored as thunk
expensiveComputation :: Int -> Int
expensiveComputation n = sum [1..n]

-- 2. Sharing: Multiple references to same thunk
shared = expensiveComputation 1000000
result1 = shared + 1  -- Creates thunk: shared + 1
result2 = shared + 2  -- Creates thunk: shared + 2

-- 3. Evaluation: Thunk computed when needed
final = result1 + result2  -- Forces evaluation of shared, result1, result2
```

### Weak Head Normal Form (WHNF): Partial Evaluation

WHNF is the evaluation state where only the outermost constructor is evaluated, not the contents.

**WHNF Examples:**
```haskell
-- WHNF for different data types
-- List: [] or (x:xs) where x and xs may be thunks
listWHNF = [1, 2, 3]  -- WHNF: (1:2:3:[])
listThunk = [1+2, 3+4, 5+6]  -- WHNF: (thunk:thunk:thunk:[])

-- Tuple: (x, y) where x and y may be thunks
tupleWHNF = (1, 2)  -- WHNF: (1, 2)
tupleThunk = (1+2, 3+4)  -- WHNF: (thunk, thunk)

-- Maybe: Nothing or Just x where x may be thunk
maybeWHNF = Just 42  -- WHNF: Just 42
maybeThunk = Just (1+2)  -- WHNF: Just thunk
```

### Call-by-Need Semantics: Evaluation on Demand

Call-by-need means expressions are evaluated at most once, and only when their results are actually needed.

**Call-by-Need in Action:**
```haskell
-- Expression evaluated at most once
expensive = verySlowComputation 1000000

-- Multiple uses of same expression
result1 = expensive + 1  -- Computes verySlowComputation
result2 = expensive + 2  -- Reuses cached result
result3 = expensive * 2  -- Reuses cached result

-- Sharing prevents recomputation
sharedResult = expensive + expensive  -- Computes once, uses twice
```

**Sharing and Memoization:**
```haskell
-- Lazy evaluation enables automatic memoization
fibs :: [Integer]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

-- Each fibonacci number computed once
fibonacci :: Int -> Integer
fibonacci n = fibs !! n

-- Usage
first10 = take 10 fibs  -- [0,1,1,2,3,5,8,13,21,34]
fib100 = fibonacci 100  -- Computed efficiently
```

## Benefits of Laziness

### Infinite Data Structures
```haskell
-- Infinite lists
naturals :: [Int]
naturals = [1..]

fibonacci :: [Int]
fibonacci = 0 : 1 : zipWith (+) fibonacci (tail fibonacci)

-- Take only what you need
first10Fibs = take 10 fibonacci
```

### Composability
```haskell
-- Fusion optimization possible
result = take 5 $ map (*2) $ filter even [1..100]
-- Can be optimized to single loop
```

### Short-Circuit Evaluation
```haskell
-- Logical operators short-circuit
safeOr :: Bool -> Bool -> Bool
safeOr True _ = True   -- Second argument not evaluated
safeOr False b = b

-- Works with infinite structures
any even [1,3,5,2,7,9..]  -- Returns True without checking all elements
```

## Potential Pitfalls

### Space Leaks
```haskell
-- BAD: Builds up large thunk
badSum :: [Int] -> Int
badSum xs = foldl (+) 0 xs

-- GOOD: Strict evaluation
goodSum :: [Int] -> Int
goodSum xs = foldl' (+) 0 xs
```

### Time Leaks
```haskell
-- Repeated computation
badLength :: [a] -> Int
badLength xs = if null xs then 0 else 1 + badLength (tail xs)

-- Each recursive call recomputes length
```

## Controlling Evaluation

### Strict Evaluation
```haskell
-- seq forces evaluation to WHNF
strictFunction :: Int -> Int -> Int
strictFunction x y = x `seq` y `seq` (x + y)

-- deepseq forces complete evaluation
import Control.DeepSeq
forceEvaluation :: NFData a => a -> a
forceEvaluation x = x `deepseq` x
```

### Bang Patterns
```haskell
{-# LANGUAGE BangPatterns #-}

-- Strict pattern matching
strictSum :: [Int] -> Int
strictSum = go 0
  where
    go !acc [] = acc
    go !acc (x:xs) = go (acc + x) xs
```

## Common Patterns

### Producer-Consumer
```haskell
-- Lazy pipeline
processData :: [String] -> [Result]
processData = map analyze . filter valid . map parse

-- Memory-efficient streaming
```

### Memoization
```haskell
-- Lazy evaluation enables memoization
fibs :: [Integer]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

-- Each fibonacci number computed once
```

## Research Papers

### Foundational Work
1. **"Lazy Evaluation" (1976)** - Peter Henderson and James H. Morris Jr.
2. **"Call-by-Need Lambda Calculus" (1995)** - John Launchbury
3. **"A Natural Semantics for Lazy Evaluation" (1993)** - John Launchbury

### Performance Studies
1. **"Measuring the Effectiveness of Lazy Evaluation" (1991)** - Paul Hudak and Mark P. Jones
2. **"Space Leaks in Haskell" (2008)** - Neil Mitchell

Lazy evaluation is a powerful feature that enables elegant solutions but requires careful consideration of performance implications.