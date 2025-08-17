# Pattern Matching and Recursion - Notes

## Overview

Pattern matching and recursion are fundamental techniques in Haskell that enable elegant and powerful programming solutions. This section explores advanced pattern matching techniques, recursion patterns, and their applications in real-world problem solving.

## Advanced Pattern Matching

### Guards vs Pattern Matching
```haskell
-- Pattern matching - structural decomposition
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Guards - conditional logic
compare' :: Ord a => a -> a -> Ordering
compare' x y
    | x < y = LT
    | x > y = GT
    | otherwise = EQ
```

### As-Patterns (@)
```haskell
-- Capture both the whole structure and parts
duplicate :: [a] -> [a]
duplicate [] = []
duplicate all@(x:xs) = x : all

-- More complex example
tails :: [a] -> [[a]]
tails [] = [[]]
tails all@(x:xs) = all : tails xs
```

### Wildcard Patterns
```haskell
-- Ignore irrelevant parts
first :: (a, b, c) -> a
first (x, _, _) = x

-- Count elements
count :: [a] -> Int
count [] = 0
count (_:xs) = 1 + count xs
```

## Recursion Patterns

### Linear Recursion
```haskell
-- Simple linear recursion
length' :: [a] -> Int
length' [] = 0
length' (_:xs) = 1 + length' xs

-- Accumulator pattern (tail recursion)
length'' :: [a] -> Int
length'' xs = go xs 0
  where
    go [] acc = acc
    go (_:ys) acc = go ys (acc + 1)
```

### Tree Recursion
```haskell
-- Multiple recursive calls
fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

-- More efficient with memoization
fibMemo :: Int -> Int
fibMemo = (map fib [0..] !!)
  where
    fib 0 = 0
    fib 1 = 1
    fib n = fibMemo (n-1) + fibMemo (n-2)
```

### Mutual Recursion
```haskell
-- Functions calling each other
even' :: Int -> Bool
even' 0 = True
even' n = odd' (n-1)

odd' :: Int -> Bool
odd' 0 = False
odd' n = even' (n-1)
```

## Data Structure Patterns

### Binary Trees
```haskell
data Tree a = Empty | Node a (Tree a) (Tree a)

-- Tree operations
insert :: Ord a => a -> Tree a -> Tree a
insert x Empty = Node x Empty Empty
insert x (Node y left right)
    | x <= y = Node y (insert x left) right
    | otherwise = Node y left (insert x right)

-- Tree traversals
inorder :: Tree a -> [a]
inorder Empty = []
inorder (Node x left right) = inorder left ++ [x] ++ inorder right
```

### Lists
```haskell
-- Advanced list patterns
takeWhile' :: (a -> Bool) -> [a] -> [a]
takeWhile' _ [] = []
takeWhile' p (x:xs)
    | p x = x : takeWhile' p xs
    | otherwise = []

-- Multiple element patterns
safeLast :: [a] -> Maybe a
safeLast [] = Nothing
safeLast [x] = Just x
safeLast (_:xs) = safeLast xs
```

## Research Papers

### Foundational Papers
1. **"Compiling Pattern Matching" (1987)** - Lennart Augustsson
2. **"The Implementation of Functional Programming Languages" (1987)** - Simon Peyton Jones
3. **"Recursion Schemes for Dynamic Programming" (2004)** - Jeremy Gibbons

### Advanced Topics
1. **"Generic Programming with Folds and Unfolds" (1999)** - Jeremy Gibbons and Geraint Jones
2. **"Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire" (1991)** - Erik Meijer

## Performance Considerations

### Tail Recursion
```haskell
-- Stack-safe recursion
sumTail :: [Int] -> Int
sumTail xs = go xs 0
  where
    go [] acc = acc
    go (y:ys) acc = go ys (acc + y)
```

### Lazy Evaluation
```haskell
-- Infinite data structures
ones :: [Int]
ones = 1 : ones

nats :: [Int]
nats = 0 : map (+1) nats
```

Pattern matching and recursion form the backbone of functional programming, enabling elegant solutions to complex problems.