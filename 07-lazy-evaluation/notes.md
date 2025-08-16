# Lazy Evaluation - Notes

## Overview

Lazy evaluation is one of Haskell's defining features, allowing computations to be deferred until their results are actually needed. This enables powerful programming patterns like infinite data structures and efficient memory usage, but requires understanding to avoid space leaks.

## How Lazy Evaluation Works

### Thunks and WHNF
```haskell
-- Unevaluated expression stored as thunk
x = 1 + 2  -- Not computed until needed

-- Weak Head Normal Form (WHNF)
-- Only evaluates to outermost constructor
```

### Call-by-Need Semantics
```haskell
-- Expression evaluated at most once
expensive = verySlowComputation
result1 = expensive + 1  -- Computes verySlowComputation
result2 = expensive + 2  -- Reuses cached result
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