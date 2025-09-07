# Pattern Matching and Recursion - Comprehensive Notes

## Overview

Pattern matching and recursion are the twin pillars of functional programming in Haskell. These fundamental techniques enable elegant, declarative, and powerful programming solutions that are both mathematically sound and practically efficient. This comprehensive guide explores advanced pattern matching techniques, recursion patterns, performance considerations, and their applications in real-world problem solving.

**Key Learning Objectives:**
- Master advanced pattern matching techniques including as-patterns, guards, and case expressions
- Understand different recursion patterns and when to use each
- Learn performance optimization techniques for recursive functions
- Apply pattern matching and recursion to solve complex real-world problems
- Understand the mathematical foundations and theoretical aspects

## Advanced Pattern Matching

Pattern matching in Haskell is a powerful mechanism that allows you to deconstruct data structures and bind variables based on their structure. It's more than just a syntax feature—it's a fundamental way of thinking about data and computation.

### Understanding Pattern Matching Fundamentals

Pattern matching works by:
1. **Structural Decomposition**: Breaking down complex data structures into their components
2. **Variable Binding**: Automatically binding variables to parts of the matched structure
3. **Conditional Execution**: Executing different code paths based on data structure shape
4. **Exhaustiveness Checking**: Ensuring all possible cases are handled (with compiler warnings)

### Guards vs Pattern Matching: When to Use What

**Pattern Matching** is ideal for:
- Structural decomposition of data types
- When the decision is based on the *shape* of data
- Destructuring complex nested structures
- When you need to bind variables to parts of the structure

```haskell
-- Pattern matching - structural decomposition
-- This is the idiomatic way to handle different cases of a data type
factorial :: Int -> Int
factorial 0 = 1                    -- Base case: exact match
factorial n = n * factorial (n - 1) -- Recursive case: bind to n

-- Pattern matching on lists
head' :: [a] -> Maybe a
head' [] = Nothing                 -- Empty list case
head' (x:_) = Just x              -- Non-empty list: bind head to x

-- Pattern matching on tuples
fst' :: (a, b) -> a
fst' (x, _) = x                   -- Bind first element, ignore second
```

**Guards** are ideal for:
- Conditional logic based on *values* rather than structure
- Complex boolean conditions
- When you need to evaluate expressions to make decisions
- Multiple conditions that don't fit pattern matching

```haskell
-- Guards - conditional logic based on values
compare' :: Ord a => a -> a -> Ordering
compare' x y
    | x < y = LT                  -- First condition
    | x > y = GT                  -- Second condition  
    | otherwise = EQ              -- Catch-all (always True)

-- Complex conditional logic
classifyAge :: Int -> String
classifyAge age
    | age < 0 = "Invalid age"
    | age < 13 = "Child"
    | age < 20 = "Teenager"
    | age < 65 = "Adult"
    | otherwise = "Senior"
```

**When to combine both:**
```haskell
-- Pattern matching + guards for complex logic
processList :: [Int] -> String
processList [] = "Empty list"                    -- Pattern match on structure
processList [x] = "Single element: " ++ show x   -- Pattern match on singleton
processList xs@(x:y:rest)                       -- As-pattern + guards
    | x > y = "First element is larger"
    | x == y = "First two elements are equal"
    | otherwise = "First element is smaller"
```

### As-Patterns (@): Capturing Structure and Parts

As-patterns are a powerful feature that allows you to capture both the entire matched structure and its individual components. The `@` symbol creates a binding to the whole structure while still allowing you to destructure it.

**Key Benefits:**
- **Efficiency**: Avoid recomputing or reconstructing data structures
- **Clarity**: Make code more readable by naming the whole structure
- **Performance**: Prevent unnecessary allocations in recursive functions

```haskell
-- Basic as-pattern: capture both whole and parts
duplicate :: [a] -> [a]
duplicate [] = []                           -- Base case: empty list
duplicate all@(x:xs) = x : all             -- all = (x:xs), so this is x : x : xs

-- More complex example: generate all tails of a list
tails :: [a] -> [[a]]
tails [] = [[]]                             -- Base case: empty list has one tail (itself)
tails all@(x:xs) = all : tails xs          -- Current list + all tails of the rest

-- Example execution:
-- tails [1,2,3] = [1,2,3] : tails [2,3]
--                = [1,2,3] : ([2,3] : tails [3])
--                = [1,2,3] : ([2,3] : ([3] : tails []))
--                = [1,2,3] : ([2,3] : ([3] : [[]]))
--                = [[1,2,3], [2,3], [3], []]
```

**Advanced As-Pattern Examples:**

```haskell
-- Tree processing with as-patterns
data Tree a = Empty | Node a (Tree a) (Tree a) deriving (Show)

-- Count nodes while preserving structure information
countNodes :: Tree a -> (Int, Tree a)
countNodes Empty = (0, Empty)
countNodes tree@(Node _ left right) = 
    let (leftCount, leftTree) = countNodes left
        (rightCount, rightTree) = countNodes right
        totalCount = 1 + leftCount + rightCount
    in (totalCount, tree)  -- Return count and original structure

-- List processing with multiple as-patterns
processPairs :: [(Int, Int)] -> [(Int, Int, Int)]
processPairs [] = []
processPairs (pair@(x, y):rest) = 
    (x, y, x + y) : processPairs rest  -- Add sum while keeping original pair

-- Complex nested as-patterns
data NestedList a = Leaf a | Branch [NestedList a] deriving (Show)

flattenWithStructure :: NestedList a -> ([a], NestedList a)
flattenWithStructure (Leaf x) = ([x], Leaf x)
flattenWithStructure tree@(Branch children) = 
    let (flattenedChildren, preservedChildren) = unzip $ map flattenWithStructure children
        allFlattened = concat flattenedChildren
    in (allFlattened, tree)  -- Return flattened list and original structure
```

**Performance Benefits of As-Patterns:**

```haskell
-- Without as-pattern (inefficient - reconstructs list)
inefficientReverse :: [a] -> [a]
inefficientReverse [] = []
inefficientReverse (x:xs) = inefficientReverse xs ++ [x]  -- O(n²) due to ++

-- With as-pattern (more efficient - reuses structure)
efficientReverse :: [a] -> [a]
efficientReverse [] = []
efficientReverse all@(x:xs) = 
    efficientReverse xs ++ [x]  -- Still O(n²), but clearer intent

-- Even better: tail-recursive version
reverseTail :: [a] -> [a]
reverseTail xs = go xs []
  where
    go [] acc = acc
    go (x:xs) acc = go xs (x:acc)  -- O(n) time, O(1) space
```

### Wildcard Patterns (_): Ignoring Irrelevant Data

Wildcard patterns (`_`) are used when you need to match a structure but don't care about the specific values in certain positions. They're essential for writing clean, focused code that only deals with the data you actually need.

**Key Benefits:**
- **Clarity**: Makes it obvious which parts of the data you're ignoring
- **Safety**: Prevents accidental use of ignored values
- **Performance**: Compiler can optimize away unused bindings
- **Maintainability**: Code is more robust to changes in data structure

```haskell
-- Basic wildcard usage: ignore irrelevant parts
first :: (a, b, c) -> a
first (x, _, _) = x                    -- Only care about first element

second :: (a, b, c) -> b  
second (_, y, _) = y                   -- Only care about second element

-- List processing: ignore head, process tail
count :: [a] -> Int
count [] = 0                           -- Base case: empty list
count (_:xs) = 1 + count xs           -- Ignore head, count tail

-- More complex wildcard patterns
processTriples :: [(Int, String, Bool)] -> [String]
processTriples [] = []
processTriples ((_, name, True):rest) = name : processTriples rest  -- Only process if Bool is True
processTriples ((_, _, False):rest) = processTriples rest           -- Skip if Bool is False
```

**Advanced Wildcard Techniques:**

```haskell
-- Nested wildcards in complex data structures
data Person = Person { name :: String, age :: Int, address :: String, phone :: String }

-- Extract only name and age, ignore address and phone
getNameAndAge :: Person -> (String, Int)
getNameAndAge (Person n a _ _) = (n, a)  -- Wildcards for unused fields

-- Pattern matching with guards and wildcards
classifyByAge :: Person -> String
classifyByAge (Person name age _ _)
    | age < 18 = name ++ " is a minor"
    | age < 65 = name ++ " is an adult"
    | otherwise = name ++ " is a senior"

-- Tree processing with wildcards
data Tree a = Empty | Node a (Tree a) (Tree a) deriving (Show)

-- Count nodes without caring about values
treeSize :: Tree a -> Int
treeSize Empty = 0
treeSize (Node _ left right) = 1 + treeSize left + treeSize right  -- Ignore node value

-- Check if tree is balanced (only care about structure, not values)
isBalanced :: Tree a -> Bool
isBalanced Empty = True
isBalanced (Node _ left right) = 
    isBalanced left && isBalanced right && 
    abs (treeHeight left - treeHeight right) <= 1
  where
    treeHeight Empty = 0
    treeHeight (Node _ l r) = 1 + max (treeHeight l) (treeHeight r)
```

**Wildcard Patterns in Case Expressions:**

```haskell
-- Complex case expression with wildcards
processCommand :: String -> String
processCommand input = case words input of
    [] -> "Empty command"
    ["help"] -> "Available commands: help, quit, echo"
    ["quit"] -> "Goodbye!"
    ("echo":_) -> "Echo command received"  -- Ignore the message
    ("set":var:_) -> "Setting variable: " ++ var  -- Ignore the value
    [cmd] -> "Unknown command: " ++ cmd
    _ -> "Invalid command format"  -- Catch-all wildcard

-- Pattern matching on Maybe with wildcards
safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x

-- Using wildcards to ignore error details
processResult :: Either String Int -> String
processResult (Right value) = "Success: " ++ show value
processResult (Left _) = "Error occurred"  -- Ignore error message
```

**Performance and Memory Considerations:**

```haskell
-- Wildcards can help with memory efficiency
-- This function only needs the length, not the actual elements
getLength :: [a] -> Int
getLength [] = 0
getLength (_:xs) = 1 + getLength xs  -- Elements are never bound, so they can be GC'd

-- Compare with this version that binds elements (potentially less efficient)
getLengthWithBinding :: [a] -> Int
getLengthWithBinding [] = 0
getLengthWithBinding (x:xs) = 1 + getLengthWithBinding xs  -- x is bound but unused

-- For large data structures, wildcards can prevent memory leaks
processLargeTree :: Tree a -> Int
processLargeTree Empty = 0
processLargeTree (Node _ left right) = 
    processLargeTree left + processLargeTree right  -- Node values never bound
```

## Recursion Patterns: The Art of Self-Reference

Recursion is the process of defining something in terms of itself. In functional programming, recursion replaces loops and provides a powerful, declarative way to solve problems. Understanding different recursion patterns is crucial for writing efficient, maintainable Haskell code.

### Understanding Recursion Fundamentals

**Key Concepts:**
- **Base Case**: The simplest case that doesn't require recursion
- **Recursive Case**: The case that calls the function with a smaller/simpler input
- **Termination**: Ensuring the recursive case eventually reaches the base case
- **Stack Usage**: How recursion uses the call stack (important for performance)

### Linear Recursion: The Foundation

Linear recursion is the simplest form where each recursive call processes one element and makes exactly one recursive call.

**Characteristics:**
- One recursive call per case
- Processes data in a linear fashion
- Stack depth equals input size
- Natural for list processing

```haskell
-- Simple linear recursion: calculating list length
length' :: [a] -> Int
length' [] = 0                    -- Base case: empty list has length 0
length' (_:xs) = 1 + length' xs   -- Recursive case: 1 + length of tail

-- Execution trace for length' [1,2,3]:
-- length' [1,2,3] = 1 + length' [2,3]
--                  = 1 + (1 + length' [3])
--                  = 1 + (1 + (1 + length' []))
--                  = 1 + (1 + (1 + 0))
--                  = 3

-- Linear recursion for list sum
sum' :: [Int] -> Int
sum' [] = 0                       -- Base case: empty list sums to 0
sum' (x:xs) = x + sum' xs         -- Recursive case: head + sum of tail

-- Linear recursion for list reversal (inefficient version)
reverse' :: [a] -> [a]
reverse' [] = []                  -- Base case: empty list reversed is empty
reverse' (x:xs) = reverse' xs ++ [x]  -- Recursive case: reverse tail + head
-- Note: This is O(n²) due to the ++ operation
```

**Accumulator Pattern (Tail Recursion):**

The accumulator pattern transforms linear recursion into tail recursion by passing an accumulator parameter that builds up the result.

**Benefits:**
- **Stack Safety**: Constant stack usage (O(1) instead of O(n))
- **Performance**: Can be optimized to a loop by the compiler
- **Memory Efficiency**: No need to build up intermediate results on the stack

```haskell
-- Accumulator pattern for length calculation
length'' :: [a] -> Int
length'' xs = go xs 0             -- Start with accumulator = 0
  where
    go [] acc = acc               -- Base case: return accumulated result
    go (_:ys) acc = go ys (acc + 1)  -- Recursive case: increment accumulator

-- Execution trace for length'' [1,2,3]:
-- go [1,2,3] 0 = go [2,3] 1
-- go [2,3] 1  = go [3] 2
-- go [3] 2    = go [] 3
-- go [] 3     = 3

-- Tail-recursive sum with accumulator
sumTail :: [Int] -> Int
sumTail xs = go xs 0
  where
    go [] acc = acc
    go (x:xs) acc = go xs (x + acc)

-- Tail-recursive reverse (efficient O(n) version)
reverseTail :: [a] -> [a]
reverseTail xs = go xs []
  where
    go [] acc = acc               -- Base case: return accumulated result
    go (x:xs) acc = go xs (x:acc)  -- Recursive case: prepend to accumulator

-- Execution trace for reverseTail [1,2,3]:
-- go [1,2,3] [] = go [2,3] [1]
-- go [2,3] [1]  = go [3] [2,1]
-- go [3] [2,1]  = go [] [3,2,1]
-- go [] [3,2,1] = [3,2,1]
```

**When to Use Each Pattern:**

```haskell
-- Use linear recursion when:
-- 1. You need to process the result after the recursive call
-- 2. The operation is naturally left-to-right
-- 3. Stack usage isn't a concern (small inputs)

-- Example: Building a list from left to right
map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs  -- Natural left-to-right processing

-- Use accumulator pattern when:
-- 1. You can build the result incrementally
-- 2. Stack usage is a concern (large inputs)
-- 3. You want maximum performance

-- Example: Folding from right to left
foldr' :: (a -> b -> b) -> b -> [a] -> b
foldr' _ acc [] = acc
foldr' f acc (x:xs) = f x (foldr' f acc xs)  -- Natural right-to-left

-- Tail-recursive version for large lists
foldrTail :: (a -> b -> b) -> b -> [a] -> b
foldrTail f acc xs = go (reverse xs) acc
  where
    go [] acc = acc
    go (x:xs) acc = go xs (f x acc)
```

### Tree Recursion: Branching into Multiple Paths

Tree recursion occurs when a function makes multiple recursive calls, creating a tree-like call structure. This pattern is common in problems that can be naturally divided into subproblems.

**Characteristics:**
- Multiple recursive calls per case
- Creates a tree of function calls
- Often leads to exponential time complexity
- Natural for divide-and-conquer algorithms

**The Classic Fibonacci Example:**

```haskell
-- Naive tree recursion: exponential time complexity O(2^n)
fibonacci :: Int -> Int
fibonacci 0 = 0                    -- Base case 1
fibonacci 1 = 1                    -- Base case 2
fibonacci n = fibonacci (n-1) + fibonacci (n-2)  -- Two recursive calls

-- Execution tree for fibonacci 4:
--                    fib(4)
--                   /      \
--              fib(3)      fib(2)
--             /      \    /      \
--        fib(2)  fib(1) fib(1) fib(0)
--       /      \
--  fib(1)  fib(0)
-- 
-- Notice: fib(2), fib(1), and fib(0) are computed multiple times!
-- This leads to exponential time complexity.

-- Time complexity analysis:
-- T(n) = T(n-1) + T(n-2) + O(1)
-- This recurrence relation has solution: T(n) = O(φ^n) where φ ≈ 1.618
-- So fibonacci 40 takes about 1 billion operations!
```

**Optimization Strategies:**

**1. Memoization (Top-Down Dynamic Programming):**

```haskell
-- Memoized version: O(n) time, O(n) space
fibMemo :: Int -> Int
fibMemo = (map fib [0..] !!)  -- Use list as memoization table
  where
    fib 0 = 0
    fib 1 = 1
    fib n = fibMemo (n-1) + fibMemo (n-2)

-- How it works:
-- 1. fibMemo n looks up the nth element in the list
-- 2. If not computed yet, it computes fib n and stores it
-- 3. Subsequent calls to fibMemo n return the cached value
-- 4. Each fibonacci number is computed exactly once

-- Alternative memoization using Data.Map
import qualified Data.Map as Map

fibMemoMap :: Int -> Int
fibMemoMap n = snd $ go n Map.empty
  where
    go 0 memo = (memo, 0)
    go 1 memo = (memo, 1)
    go n memo = 
        case Map.lookup n memo of
            Just result -> (memo, result)
            Nothing -> 
                let (memo1, fib1) = go (n-1) memo
                    (memo2, fib2) = go (n-2) memo1
                    result = fib1 + fib2
                    newMemo = Map.insert n result memo2
                in (newMemo, result)
```

**2. Bottom-Up Dynamic Programming:**

```haskell
-- Bottom-up approach: O(n) time, O(1) space
fibBottomUp :: Int -> Int
fibBottomUp n
    | n < 0 = error "Negative input"
    | n == 0 = 0
    | n == 1 = 1
    | otherwise = go 2 0 1 n
  where
    go i prev curr target
        | i > target = curr
        | otherwise = go (i+1) curr (prev + curr) target

-- Execution for fibBottomUp 5:
-- go 2 0 1 5 = go 3 1 1 5    (prev=0, curr=1)
-- go 3 1 1 5 = go 4 1 2 5    (prev=1, curr=1)
-- go 4 1 2 5 = go 5 2 3 5    (prev=1, curr=2)
-- go 5 2 3 5 = go 6 3 5 5    (prev=2, curr=3)
-- go 6 3 5 5 = 5              (i > target, return curr=5)
```

**3. Matrix Exponentiation (Advanced):**

```haskell
-- O(log n) time using matrix exponentiation
-- Based on the identity: [F(n+1)]   [1 1]^n [F(1)]
--                        [F(n)  ] = [1 0]   [F(0)]

fibMatrix :: Int -> Int
fibMatrix n
    | n < 0 = error "Negative input"
    | otherwise = snd $ matrixPower (1, 1, 1, 0) n

-- Matrix multiplication for 2x2 matrices
matrixMult :: (Int, Int, Int, Int) -> (Int, Int, Int, Int) -> (Int, Int, Int, Int)
matrixMult (a, b, c, d) (e, f, g, h) = 
    (a*e + b*g, a*f + b*h, c*e + d*g, c*f + d*h)

-- Matrix exponentiation using binary exponentiation
matrixPower :: (Int, Int, Int, Int) -> Int -> (Int, Int, Int, Int)
matrixPower _ 0 = (1, 0, 0, 1)  -- Identity matrix
matrixPower m 1 = m
matrixPower m n
    | even n = let half = matrixPower m (n `div` 2)
               in matrixMult half half
    | otherwise = let half = matrixPower m ((n-1) `div` 2)
                  in matrixMult m (matrixMult half half)
```

**Real-World Tree Recursion Examples:**

```haskell
-- Binary tree operations
data Tree a = Empty | Node a (Tree a) (Tree a) deriving (Show)

-- Tree size: O(n) time, O(h) space where h is height
treeSize :: Tree a -> Int
treeSize Empty = 0
treeSize (Node _ left right) = 1 + treeSize left + treeSize right

-- Tree height: O(n) time, O(h) space
treeHeight :: Tree a -> Int
treeHeight Empty = 0
treeHeight (Node _ left right) = 1 + max (treeHeight left) (treeHeight right)

-- Binary search tree insertion
insertBST :: Ord a => a -> Tree a -> Tree a
insertBST x Empty = Node x Empty Empty
insertBST x (Node y left right)
    | x <= y = Node y (insertBST x left) right
    | otherwise = Node y left (insertBST x right)

-- Quicksort: divide and conquer
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (pivot:rest) = 
    quicksort smaller ++ [pivot] ++ quicksort larger
  where
    smaller = [x | x <- rest, x <= pivot]
    larger = [x | x <- rest, x > pivot]

-- Time complexity: O(n log n) average case, O(n²) worst case
-- Space complexity: O(log n) average case, O(n) worst case
```

**Performance Analysis of Tree Recursion:**

```haskell
-- Example: Computing all paths in a binary tree
data Tree a = Empty | Node a (Tree a) (Tree a) deriving (Show)

-- Naive approach: exponential time
allPaths :: Tree a -> [[a]]
allPaths Empty = []
allPaths (Node x Empty Empty) = [[x]]
allPaths (Node x left right) = 
    map (x:) (allPaths left ++ allPaths right)

-- Optimized approach: linear time
allPathsOptimized :: Tree a -> [[a]]
allPathsOptimized tree = go tree []
  where
    go Empty _ = []
    go (Node x Empty Empty) path = [reverse (x:path)]
    go (Node x left right) path = 
        go left (x:path) ++ go right (x:path)

-- The key insight: build the path incrementally instead of
-- reconstructing it at each step
```

### Mutual Recursion: Functions That Call Each Other

Mutual recursion occurs when two or more functions call each other in a circular fashion. This pattern is useful for problems that can be naturally divided into multiple related subproblems.

**Characteristics:**
- Functions call each other directly or indirectly
- Often used for parsing, state machines, and complex algorithms
- Can be more natural than single-function recursion for certain problems
- Requires careful design to ensure termination

**Basic Mutual Recursion Example:**

```haskell
-- Classic example: even and odd functions
even' :: Int -> Bool
even' 0 = True                    -- Base case: 0 is even
even' n = odd' (n-1)             -- Recursive case: n is even if n-1 is odd

odd' :: Int -> Bool
odd' 0 = False                   -- Base case: 0 is not odd
odd' n = even' (n-1)             -- Recursive case: n is odd if n-1 is even

-- Execution trace for even' 4:
-- even' 4 = odd' 3
-- odd' 3  = even' 2
-- even' 2 = odd' 1
-- odd' 1  = even' 0
-- even' 0 = True
-- So even' 4 = True

-- More efficient version using modulo
evenEfficient :: Int -> Bool
evenEfficient n = n `mod` 2 == 0

oddEfficient :: Int -> Bool
oddEfficient n = n `mod` 2 == 1
```

**Advanced Mutual Recursion Examples:**

**1. Parser with Mutual Recursion:**

```haskell
-- Expression parser with operator precedence
data Expr = Num Int | Add Expr Expr | Mul Expr Expr deriving (Show)

-- Parse expression (lowest precedence: addition/subtraction)
parseExpr :: String -> Maybe (Expr, String)
parseExpr input = parseTerm input

-- Parse term (higher precedence: multiplication/division)
parseTerm :: String -> Maybe (Expr, String)
parseTerm input = do
    (left, rest) <- parseFactor input
    parseTermRest left rest
  where
    parseTermRest left ('*':rest) = do
        (right, rest2) <- parseFactor rest
        parseTermRest (Mul left right) rest2
    parseTermRest left ('/':rest) = do
        (right, rest2) <- parseFactor rest
        parseTermRest (Mul left right) rest2  -- Simplified: treat / as *
    parseTermRest left rest = Just (left, rest)

-- Parse factor (highest precedence: numbers and parentheses)
parseFactor :: String -> Maybe (Expr, String)
parseFactor ('(':rest) = do
    (expr, ')':rest2) <- parseExpr rest
    Just (expr, rest2)
parseFactor input = parseNumber input

-- Parse number
parseNumber :: String -> Maybe (Expr, String)
parseNumber input = 
    let (numStr, rest) = span isDigit input
    in if null numStr 
       then Nothing
       else Just (Num (read numStr), rest)

-- Example: parseExpr "2+3*4" = Just (Add (Num 2) (Mul (Num 3) (Num 4)), "")
```

**2. State Machine with Mutual Recursion:**

```haskell
-- Simple state machine for processing commands
data State = Idle | Processing | Error deriving (Eq, Show)
data Command = Start | Process | Stop | Reset deriving (Show)

-- State transition functions
idleState :: Command -> (State, String)
idleState Start = (Processing, "Started processing")
idleState Process = (Error, "Cannot process in idle state")
idleState Stop = (Idle, "Already idle")
idleState Reset = (Idle, "Reset to idle")

processingState :: Command -> (State, String)
processingState Start = (Processing, "Already processing")
processingState Process = (Processing, "Processing...")
processingState Stop = (Idle, "Stopped processing")
processingState Reset = (Idle, "Reset from processing")

errorState :: Command -> (State, String)
errorState Start = (Error, "Cannot start from error state")
errorState Process = (Error, "Cannot process in error state")
errorState Stop = (Error, "Cannot stop from error state")
errorState Reset = (Idle, "Reset from error")

-- Main state machine function
stateMachine :: State -> [Command] -> [(State, String)]
stateMachine _ [] = []
stateMachine state (cmd:cmds) = 
    let (newState, message) = transition state cmd
    in (newState, message) : stateMachine newState cmds
  where
    transition Idle = idleState
    transition Processing = processingState
    transition Error = errorState

-- Example usage:
-- stateMachine Idle [Start, Process, Process, Stop, Reset]
-- = [(Processing, "Started processing"), (Processing, "Processing..."), 
--    (Processing, "Processing..."), (Idle, "Stopped processing"), (Idle, "Reset to idle")]
```

**3. Tree Traversal with Mutual Recursion:**

```haskell
-- Binary tree with different traversal strategies
data Tree a = Empty | Node a (Tree a) (Tree a) deriving (Show)

-- Preorder traversal: root, left, right
preorder :: Tree a -> [a]
preorder Empty = []
preorder (Node x left right) = x : (preorder left ++ preorder right)

-- Inorder traversal: left, root, right
inorder :: Tree a -> [a]
inorder Empty = []
inorder (Node x left right) = inorder left ++ [x] ++ inorder right

-- Postorder traversal: left, right, root
postorder :: Tree a -> [a]
postorder Empty = []
postorder (Node x left right) = postorder left ++ postorder right ++ [x]

-- Mutual recursion for complex tree operations
-- Check if tree is balanced and return height
isBalancedWithHeight :: Tree a -> (Bool, Int)
isBalancedWithHeight Empty = (True, 0)
isBalancedWithHeight (Node _ left right) = 
    let (leftBalanced, leftHeight) = isBalancedWithHeight left
        (rightBalanced, rightHeight) = isBalancedWithHeight right
        heightDiff = abs (leftHeight - rightHeight)
        isBalanced = leftBalanced && rightBalanced && heightDiff <= 1
        height = 1 + max leftHeight rightHeight
    in (isBalanced, height)

-- Check if tree is a valid BST
isValidBST :: Ord a => Tree a -> Bool
isValidBST Empty = True
isValidBST (Node x left right) = 
    isBSTNode x left right && isValidBST left && isValidBST right
  where
    isBSTNode val Empty Empty = True
    isBSTNode val Empty (Node rVal _ _) = val < rVal
    isBSTNode val (Node lVal _ _) Empty = val > lVal
    isBSTNode val (Node lVal _ _) (Node rVal _ _) = val > lVal && val < rVal
```

**4. Graph Traversal with Mutual Recursion:**

```haskell
-- Simple graph representation
type Graph = [(Int, [Int])]  -- (node, neighbors)

-- DFS with mutual recursion for visited tracking
dfs :: Graph -> Int -> [Int]
dfs graph start = dfsHelper graph start []

dfsHelper :: Graph -> Int -> [Int] -> [Int]
dfsHelper graph node visited
    | node `elem` visited = visited  -- Already visited
    | otherwise = 
        let neighbors = case lookup node graph of
                         Just ns -> ns
                         Nothing -> []
            newVisited = node : visited
        in foldl (dfsHelper graph) newVisited neighbors

-- BFS using mutual recursion
bfs :: Graph -> Int -> [Int]
bfs graph start = bfsHelper graph [start] []

bfsHelper :: Graph -> [Int] -> [Int] -> [Int]
bfsHelper _ [] visited = reverse visited
bfsHelper graph (current:queue) visited
    | current `elem` visited = bfsHelper graph queue visited
    | otherwise = 
        let neighbors = case lookup current graph of
                         Just ns -> ns
                         Nothing -> []
            newVisited = current : visited
            newQueue = queue ++ neighbors
        in bfsHelper graph newQueue newVisited
```

**Performance Considerations for Mutual Recursion:**

```haskell
-- Mutual recursion can sometimes be optimized by combining functions
-- Example: Instead of separate even/odd functions, use a single function

-- Inefficient mutual recursion
evenOdd :: Int -> (Bool, Bool)
evenOdd 0 = (True, False)
evenOdd n = let (e, o) = evenOdd (n-1) in (o, e)

-- More efficient single function
evenOddEfficient :: Int -> (Bool, Bool)
evenOddEfficient n = (n `mod` 2 == 0, n `mod` 2 == 1)

-- For complex mutual recursion, consider using memoization
import qualified Data.Map as Map

-- Memoized mutual recursion
fibPair :: Int -> (Int, Int)
fibPair 0 = (0, 1)
fibPair n = 
    let (fibN, fibNPlus1) = fibPair (n-1)
    in (fibNPlus1, fibN + fibNPlus1)

-- With memoization
fibPairMemo :: Int -> (Int, Int)
fibPairMemo n = snd $ go n Map.empty
  where
    go 0 memo = (memo, (0, 1))
    go n memo = 
        case Map.lookup n memo of
            Just result -> (memo, result)
            Nothing -> 
                let (memo1, (fibN, fibNPlus1)) = go (n-1) memo
                    result = (fibNPlus1, fibN + fibNPlus1)
                    newMemo = Map.insert n result memo1
                in (newMemo, result)
```

**When to Use Mutual Recursion:**

1. **Natural Problem Division**: When a problem naturally splits into multiple related subproblems
2. **State Machines**: When modeling systems with multiple states
3. **Parsing**: When dealing with grammars with multiple non-terminals
4. **Complex Algorithms**: When the algorithm has multiple phases that call each other

**When to Avoid Mutual Recursion:**

1. **Simple Problems**: When a single recursive function is sufficient
2. **Performance Critical**: When the overhead of multiple function calls matters
3. **Stack Overflow Risk**: When deep mutual recursion could cause stack overflow
4. **Complexity**: When it makes the code harder to understand than alternatives

## Data Structure Patterns: Recursive Data Types

Recursive data types are the foundation of functional programming. They allow us to represent complex, hierarchical structures using simple, elegant definitions. Understanding how to work with these structures using pattern matching and recursion is crucial for effective Haskell programming.

### Binary Trees: The Classic Recursive Structure

Binary trees are perhaps the most fundamental recursive data structure. They demonstrate how pattern matching and recursion work together to process hierarchical data.

**Tree Definition and Basic Operations:**

```haskell
-- Basic binary tree definition
data Tree a = Empty | Node a (Tree a) (Tree a) deriving (Show, Eq)

-- Tree size: count all nodes
treeSize :: Tree a -> Int
treeSize Empty = 0
treeSize (Node _ left right) = 1 + treeSize left + treeSize right

-- Tree height: maximum depth
treeHeight :: Tree a -> Int
treeHeight Empty = 0
treeHeight (Node _ left right) = 1 + max (treeHeight left) (treeHeight right)

-- Check if tree is empty
isEmpty :: Tree a -> Bool
isEmpty Empty = True
isEmpty _ = False

-- Get root value (unsafe)
rootValue :: Tree a -> a
rootValue (Node x _ _) = x
rootValue Empty = error "Cannot get root of empty tree"

-- Safe root value extraction
safeRootValue :: Tree a -> Maybe a
safeRootValue Empty = Nothing
safeRootValue (Node x _ _) = Just x
```

**Tree Traversals: Different Ways to Visit Nodes**

```haskell
-- Preorder traversal: Root, Left, Right
preorder :: Tree a -> [a]
preorder Empty = []
preorder (Node x left right) = x : (preorder left ++ preorder right)

-- Inorder traversal: Left, Root, Right (gives sorted order for BST)
inorder :: Tree a -> [a]
inorder Empty = []
inorder (Node x left right) = inorder left ++ [x] ++ inorder right

-- Postorder traversal: Left, Right, Root
postorder :: Tree a -> [a]
postorder Empty = []
postorder (Node x left right) = postorder left ++ postorder right ++ [x]

-- Level-order traversal (breadth-first)
levelOrder :: Tree a -> [a]
levelOrder Empty = []
levelOrder tree = go [tree]
  where
    go [] = []
    go (Empty:rest) = go rest
    go (Node x left right:rest) = x : go (rest ++ [left, right])

-- Example tree:       1
--                   /   \
--                  2     3
--                 / \   / \
--                4   5 6   7
-- 
-- preorder:  [1,2,4,5,3,6,7]
-- inorder:   [4,2,5,1,6,3,7]
-- postorder: [4,5,2,6,7,3,1]
-- levelorder:[1,2,3,4,5,6,7]
```

**Binary Search Tree Operations:**

```haskell
-- Insert into BST
insert :: Ord a => a -> Tree a -> Tree a
insert x Empty = Node x Empty Empty
insert x (Node y left right)
    | x <= y = Node y (insert x left) right
    | otherwise = Node y left (insert x right)

-- Search in BST
search :: Ord a => a -> Tree a -> Bool
search _ Empty = False
search x (Node y left right)
    | x == y = True
    | x < y = search x left
    | otherwise = search x right

-- Find minimum value in BST
findMin :: Tree a -> Maybe a
findMin Empty = Nothing
findMin (Node x Empty _) = Just x
findMin (Node _ left _) = findMin left

-- Find maximum value in BST
findMax :: Tree a -> Maybe a
findMax Empty = Nothing
findMax (Node x _ Empty) = Just x
findMax (Node _ _ right) = findMax right

-- Delete from BST (complex operation)
delete :: Ord a => a -> Tree a -> Tree a
delete _ Empty = Empty
delete x (Node y left right)
    | x < y = Node y (delete x left) right
    | x > y = Node y left (delete x right)
    | otherwise = deleteRoot (Node y left right)
  where
    deleteRoot (Node _ Empty right) = right
    deleteRoot (Node _ left Empty) = left
    deleteRoot (Node _ left right) = 
        let Just minVal = findMin right
        in Node minVal left (delete minVal right)

-- Check if tree is valid BST
isValidBST :: Ord a => Tree a -> Bool
isValidBST Empty = True
isValidBST (Node x left right) = 
    all (< x) (inorder left) && 
    all (> x) (inorder right) && 
    isValidBST left && 
    isValidBST right

-- More efficient BST validation
isValidBSTEfficient :: Ord a => Tree a -> Bool
isValidBSTEfficient tree = go tree Nothing Nothing
  where
    go Empty _ _ = True
    go (Node x left right) minVal maxVal =
        let validMin = case minVal of
                        Nothing -> True
                        Just m -> x > m
            validMax = case maxVal of
                        Nothing -> True
                        Just m -> x < m
        in validMin && validMax && 
           go left minVal (Just x) && 
           go right (Just x) maxVal
```

**Advanced Tree Operations:**

```haskell
-- Mirror/flip tree
mirror :: Tree a -> Tree a
mirror Empty = Empty
mirror (Node x left right) = Node x (mirror right) (mirror left)

-- Check if two trees are identical
identical :: Eq a => Tree a -> Tree a -> Bool
identical Empty Empty = True
identical (Node x1 l1 r1) (Node x2 l2 r2) = 
    x1 == x2 && identical l1 l2 && identical r1 r2
identical _ _ = False

-- Check if tree is balanced (height difference <= 1)
isBalanced :: Tree a -> Bool
isBalanced Empty = True
isBalanced (Node _ left right) = 
    isBalanced left && isBalanced right &&
    abs (treeHeight left - treeHeight right) <= 1

-- More efficient balanced check (single pass)
isBalancedEfficient :: Tree a -> Bool
isBalancedEfficient tree = fst (checkBalanced tree)
  where
    checkBalanced Empty = (True, 0)
    checkBalanced (Node _ left right) = 
        let (leftBalanced, leftHeight) = checkBalanced left
            (rightBalanced, rightHeight) = checkBalanced right
            heightDiff = abs (leftHeight - rightHeight)
            isBalanced = leftBalanced && rightBalanced && heightDiff <= 1
            height = 1 + max leftHeight rightHeight
        in (isBalanced, height)

-- Convert tree to list using different strategies
toListPreorder :: Tree a -> [a]
toListPreorder = preorder

toListInorder :: Tree a -> [a]
toListInorder = inorder

toListPostorder :: Tree a -> [a]
toListPostorder = postorder

-- Build tree from list (creates right-skewed tree)
fromList :: [a] -> Tree a
fromList = foldr insert Empty

-- Build balanced tree from sorted list
fromSortedList :: [a] -> Tree a
fromSortedList [] = Empty
fromSortedList xs = 
    let mid = length xs `div` 2
        (left, x:right) = splitAt mid xs
    in Node x (fromSortedList left) (fromSortedList right)
```

**Performance Analysis of Tree Operations:**

```haskell
-- Time complexity analysis for BST operations:
-- 
-- Search: O(h) where h is height
--   - Best case (balanced): O(log n)
--   - Worst case (skewed): O(n)
-- 
-- Insert: O(h) where h is height
--   - Best case (balanced): O(log n)
--   - Worst case (skewed): O(n)
-- 
-- Delete: O(h) where h is height
--   - Best case (balanced): O(log n)
--   - Worst case (skewed): O(n)
-- 
-- Traversals: O(n) - must visit every node
-- 
-- Space complexity:
-- - Tree storage: O(n)
-- - Recursion stack: O(h) where h is height
--   - Best case (balanced): O(log n)
--   - Worst case (skewed): O(n)

-- Example: Building a balanced BST
buildBalancedBST :: [Int] -> Tree Int
buildBalancedBST xs = fromSortedList (sort xs)
  where
    sort [] = []
    sort (x:xs) = sort [y | y <- xs, y <= x] ++ [x] ++ sort [y | y <- xs, y > x]

-- This ensures O(log n) height, giving us O(log n) search/insert/delete
```

### Lists: The Workhorse of Functional Programming

Lists are the most commonly used recursive data structure in Haskell. They demonstrate many important patterns and techniques that apply to other recursive structures.

**Advanced List Pattern Matching:**

```haskell
-- Basic list patterns
head' :: [a] -> Maybe a
head' [] = Nothing
head' (x:_) = Just x

tail' :: [a] -> Maybe [a]
tail' [] = Nothing
tail' (_:xs) = Just xs

-- Multiple element patterns
safeLast :: [a] -> Maybe a
safeLast [] = Nothing
safeLast [x] = Just x
safeLast (_:xs) = safeLast xs

-- Pattern matching on specific lengths
describeList :: [a] -> String
describeList [] = "Empty list"
describeList [_] = "Singleton list"
describeList [_, _] = "List with two elements"
describeList [_, _, _] = "List with three elements"
describeList _ = "List with many elements"

-- Complex list patterns with guards
takeWhile' :: (a -> Bool) -> [a] -> [a]
takeWhile' _ [] = []
takeWhile' p (x:xs)
    | p x = x : takeWhile' p xs
    | otherwise = []

-- Drop while with pattern matching
dropWhile' :: (a -> Bool) -> [a] -> [a]
dropWhile' _ [] = []
dropWhile' p xs@(x:rest)
    | p x = dropWhile' p rest
    | otherwise = xs

-- Split at first occurrence
splitAtFirst :: Eq a => a -> [a] -> ([a], [a])
splitAtFirst _ [] = ([], [])
splitAtFirst y (x:xs)
    | x == y = ([], xs)
    | otherwise = let (before, after) = splitAtFirst y xs
                  in (x:before, after)
```

**List Processing Patterns:**

```haskell
-- Map with pattern matching
map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs

-- Filter with pattern matching
filter' :: (a -> Bool) -> [a] -> [a]
filter' _ [] = []
filter' p (x:xs)
    | p x = x : filter' p xs
    | otherwise = filter' p xs

-- Fold with pattern matching
foldr' :: (a -> b -> b) -> b -> [a] -> b
foldr' _ acc [] = acc
foldr' f acc (x:xs) = f x (foldr' f acc xs)

foldl' :: (b -> a -> b) -> b -> [a] -> b
foldl' _ acc [] = acc
foldl' f acc (x:xs) = foldl' f (f acc x) xs

-- Zip with pattern matching
zip' :: [a] -> [b] -> [(a, b)]
zip' [] _ = []
zip' _ [] = []
zip' (x:xs) (y:ys) = (x, y) : zip' xs ys

-- Unzip with pattern matching
unzip' :: [(a, b)] -> ([a], [b])
unzip' [] = ([], [])
unzip' ((x, y):rest) = 
    let (xs, ys) = unzip' rest
    in (x:xs, y:ys)
```

**Advanced List Operations:**

```haskell
-- Partition list based on predicate
partition' :: (a -> Bool) -> [a] -> ([a], [a])
partition' _ [] = ([], [])
partition' p (x:xs) = 
    let (trues, falses) = partition' p xs
    in if p x then (x:trues, falses) else (trues, x:falses)

-- Group consecutive equal elements
group' :: Eq a => [a] -> [[a]]
group' [] = []
group' (x:xs) = (x:takeWhile (== x) xs) : group' (dropWhile (== x) xs)

-- Split list into chunks of given size
chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

-- Rotate list left by n positions
rotateLeft :: Int -> [a] -> [a]
rotateLeft n xs = drop n xs ++ take n xs

-- Rotate list right by n positions
rotateRight :: Int -> [a] -> [a]
rotateRight n xs = drop (length xs - n) xs ++ take (length xs - n) xs

-- Find all sublists
sublists :: [a] -> [[a]]
sublists [] = [[]]
sublists (x:xs) = sublists xs ++ map (x:) (sublists xs)

-- Find all contiguous sublists
contiguousSublists :: [a] -> [[a]]
contiguousSublists [] = []
contiguousSublists xs = tails xs ++ contiguousSublists (tail xs)
  where
    tails [] = []
    tails (x:xs) = [x] : map (x:) (tails xs)
```

**List Performance Analysis:**

```haskell
-- Time complexity analysis for common list operations:
--
-- head, tail: O(1) - constant time
-- length: O(n) - must traverse entire list
-- last: O(n) - must traverse entire list
-- (!!): O(n) - must traverse to index
-- reverse: O(n) - must traverse entire list
-- ++ (append): O(n) where n is length of first list
-- concat: O(m) where m is total number of elements
-- map: O(n) - must apply function to each element
-- filter: O(n) - must check each element
-- foldr/foldl: O(n) - must process each element
--
-- Space complexity:
-- - List storage: O(n) where n is number of elements
-- - Recursion stack: O(n) for non-tail-recursive functions
-- - Tail-recursive functions: O(1) stack space

-- Example: Inefficient vs Efficient list operations

-- Inefficient: O(n²) due to repeated ++ operations
reverseInefficient :: [a] -> [a]
reverseInefficient [] = []
reverseInefficient (x:xs) = reverseInefficient xs ++ [x]

-- Efficient: O(n) using accumulator pattern
reverseEfficient :: [a] -> [a]
reverseEfficient xs = go xs []
  where
    go [] acc = acc
    go (x:xs) acc = go xs (x:acc)

-- Inefficient: O(n²) due to repeated ++ operations
concatInefficient :: [[a]] -> [a]
concatInefficient [] = []
concatInefficient (xs:xss) = xs ++ concatInefficient xss

-- Efficient: O(n) using foldr
concatEfficient :: [[a]] -> [a]
concatEfficient = foldr (++) []

-- Inefficient: O(n²) due to repeated length calls
middleInefficient :: [a] -> Maybe a
middleInefficient [] = Nothing
middleInefficient xs = 
    let len = length xs
        mid = len `div` 2
    in Just (xs !! mid)

-- Efficient: O(n) using two pointers
middleEfficient :: [a] -> Maybe a
middleEfficient [] = Nothing
middleEfficient xs = Just (fst (go xs xs))
  where
    go (_:xs) (_:_:ys) = go xs ys  -- Fast pointer moves 2 steps
    go (x:_) _ = (x, [])           -- Slow pointer moves 1 step
    go [] _ = error "Impossible"   -- Should never happen
```

**List Comprehensions and Pattern Matching:**

```haskell
-- List comprehensions with pattern matching
-- Extract first elements from list of pairs
firsts :: [(a, b)] -> [a]
firsts pairs = [x | (x, _) <- pairs]

-- Extract elements at even indices
evenIndices :: [a] -> [a]
evenIndices xs = [x | (x, i) <- zip xs [0..], even i]

-- Find all pairs that sum to target
pairsWithSum :: Int -> [Int] -> [(Int, Int)]
pairsWithSum target xs = 
    [(x, y) | x <- xs, y <- xs, x + y == target, x <= y]

-- Generate all permutations (inefficient but educational)
permutations :: [a] -> [[a]]
permutations [] = [[]]
permutations xs = 
    [x:perm | x <- xs, perm <- permutations (xs \\ [x])]
  where
    (\\) :: Eq a => [a] -> [a] -> [a]
    (\\) [] _ = []
    (\\) (x:xs) ys
        | x `elem` ys = xs \\ ys
        | otherwise = x : (xs \\ ys)
```

**Real-World List Processing Examples:**

```haskell
-- CSV parsing with pattern matching
parseCSV :: String -> [[String]]
parseCSV input = map parseRow (lines input)
  where
    parseRow row = splitOn ',' row
    splitOn _ [] = []
    splitOn delim str = 
        let (before, after) = break (== delim) str
        in before : if null after then [] else splitOn delim (tail after)

-- Text processing with pattern matching
wordCount :: String -> [(String, Int)]
wordCount text = 
    let words = map (filter isAlpha) (splitOn ' ' (map toLower text))
        grouped = group' (sort words)
    in map (\ws -> (head ws, length ws)) grouped

-- Data validation with pattern matching
validateEmail :: String -> Bool
validateEmail email = 
    case splitOn '@' email of
        [local, domain] -> 
            not (null local) && 
            not (null domain) && 
            '@' `notElem` local && 
            '.' `elem` domain
        _ -> False

-- Financial calculations
calculateCompoundInterest :: Double -> Double -> Int -> Double
calculateCompoundInterest principal rate years = 
    principal * (1 + rate) ^ years

-- Portfolio analysis
portfolioValue :: [(String, Double, Double)] -> Double
portfolioValue holdings = 
    sum [shares * price | (_, shares, price) <- holdings]

-- Risk analysis
portfolioRisk :: [(String, Double, Double, Double)] -> Double
portfolioRisk holdings = 
    sqrt $ sum [weight * weight * variance | (_, weight, _, variance) <- holdings]
  where
    totalWeight = sum [weight | (_, weight, _, _) <- holdings]
    normalizedHoldings = [(name, weight/totalWeight, price, variance) | 
                         (name, weight, price, variance) <- holdings]
```

## Performance Analysis and Optimization

Understanding the performance characteristics of pattern matching and recursion is crucial for writing efficient Haskell programs. This section covers optimization techniques, space/time complexity analysis, and best practices.

### Time Complexity Analysis

**Pattern Matching Performance:**
- **Constant Time O(1)**: Simple pattern matching on constructors
- **Linear Time O(n)**: Pattern matching that requires traversing data structures
- **Exponential Time O(2^n)**: Tree recursion without optimization

```haskell
-- O(1) - constant time pattern matching
head' :: [a] -> Maybe a
head' [] = Nothing
head' (x:_) = Just x

-- O(n) - linear time due to list traversal
last' :: [a] -> Maybe a
last' [] = Nothing
last' [x] = Just x
last' (_:xs) = last' xs

-- O(2^n) - exponential time (naive Fibonacci)
fibNaive :: Int -> Int
fibNaive 0 = 0
fibNaive 1 = 1
fibNaive n = fibNaive (n-1) + fibNaive (n-2)

-- O(n) - linear time with memoization
fibMemo :: Int -> Int
fibMemo = (map fib [0..] !!)
  where
    fib 0 = 0
    fib 1 = 1
    fib n = fibMemo (n-1) + fibMemo (n-2)
```

**Recursion Performance Patterns:**

```haskell
-- Linear recursion: O(n) time, O(n) space
length' :: [a] -> Int
length' [] = 0
length' (_:xs) = 1 + length' xs

-- Tail recursion: O(n) time, O(1) space
lengthTail :: [a] -> Int
lengthTail xs = go xs 0
  where
    go [] acc = acc
    go (_:ys) acc = go ys (acc + 1)

-- Tree recursion: O(2^n) time, O(n) space
fibTree :: Int -> Int
fibTree 0 = 0
fibTree 1 = 1
fibTree n = fibTree (n-1) + fibTree (n-2)

-- Optimized tree recursion: O(n) time, O(n) space
fibOptimized :: Int -> Int
fibOptimized n = snd $ go n (0, 1)
  where
    go 0 (a, b) = (a, b)
    go n (a, b) = go (n-1) (b, a + b)
```

### Space Complexity Analysis

**Stack Usage Patterns:**

```haskell
-- High stack usage: O(n) stack space
reverseStack :: [a] -> [a]
reverseStack [] = []
reverseStack (x:xs) = reverseStack xs ++ [x]

-- Low stack usage: O(1) stack space (tail recursive)
reverseTail :: [a] -> [a]
reverseTail xs = go xs []
  where
    go [] acc = acc
    go (x:xs) acc = go xs (x:acc)

-- Memory usage analysis
-- reverseStack [1,2,3,4,5]:
--   reverseStack [1,2,3,4,5] = reverseStack [2,3,4,5] ++ [1]
--   reverseStack [2,3,4,5] = reverseStack [3,4,5] ++ [2]
--   reverseStack [3,4,5] = reverseStack [4,5] ++ [3]
--   reverseStack [4,5] = reverseStack [5] ++ [4]
--   reverseStack [5] = reverseStack [] ++ [5]
--   reverseStack [] = []
--   Stack depth: 5 (O(n))

-- reverseTail [1,2,3,4,5]:
--   go [1,2,3,4,5] [] = go [2,3,4,5] [1]
--   go [2,3,4,5] [1] = go [3,4,5] [2,1]
--   go [3,4,5] [2,1] = go [4,5] [3,2,1]
--   go [4,5] [3,2,1] = go [5] [4,3,2,1]
--   go [5] [4,3,2,1] = go [] [5,4,3,2,1]
--   go [] [5,4,3,2,1] = [5,4,3,2,1]
--   Stack depth: 1 (O(1))
```

### Optimization Techniques

**1. Tail Recursion Optimization:**

```haskell
-- Convert non-tail-recursive to tail-recursive
-- Original: O(n) stack space
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Optimized: O(1) stack space
factorialTail :: Int -> Int
factorialTail n = go n 1
  where
    go 0 acc = acc
    go n acc = go (n - 1) (n * acc)

-- General pattern for tail recursion
-- Original: f x = g (f (h x))
-- Optimized: f x = go x initialAcc
--           where go x acc = if baseCase x then acc else go (h x) (g acc)
```

**2. Memoization Techniques:**

```haskell
-- Simple memoization using lazy evaluation
fibMemoLazy :: Int -> Int
fibMemoLazy = (map fib [0..] !!)
  where
    fib 0 = 0
    fib 1 = 1
    fib n = fibMemoLazy (n-1) + fibMemoLazy (n-2)

-- Memoization with explicit cache
import qualified Data.Map as Map

fibMemoMap :: Int -> Int
fibMemoMap n = snd $ go n Map.empty
  where
    go 0 memo = (memo, 0)
    go 1 memo = (memo, 1)
    go n memo = 
        case Map.lookup n memo of
            Just result -> (memo, result)
            Nothing -> 
                let (memo1, fib1) = go (n-1) memo
                    (memo2, fib2) = go (n-2) memo1
                    result = fib1 + fib2
                    newMemo = Map.insert n result memo2
                in (newMemo, result)

-- Memoization with array (most efficient for dense lookups)
import Data.Array

fibMemoArray :: Int -> Int
fibMemoArray n = fibArray ! n
  where
    fibArray = array (0, n) [(i, fib i) | i <- [0..n]]
    fib 0 = 0
    fib 1 = 1
    fib i = fibArray ! (i-1) + fibArray ! (i-2)
```

**3. Accumulator Pattern:**

```haskell
-- Convert recursive functions to use accumulators
-- Original: builds result on the way back
reverseRecursive :: [a] -> [a]
reverseRecursive [] = []
reverseRecursive (x:xs) = reverseRecursive xs ++ [x]

-- With accumulator: builds result on the way down
reverseAccumulator :: [a] -> [a]
reverseAccumulator xs = go xs []
  where
    go [] acc = acc
    go (x:xs) acc = go xs (x:acc)

-- General accumulator pattern
-- Original: f x = if baseCase x then baseValue else g (f (h x))
-- With accumulator: f x = go x initialAcc
--                  where go x acc = if baseCase x then acc else go (h x) (g acc)
```

**4. Lazy Evaluation Optimization:**

```haskell
-- Lazy evaluation can prevent unnecessary computation
-- This only computes as much as needed
fibs :: [Int]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

-- Get nth Fibonacci number (only computes up to n)
nthFib :: Int -> Int
nthFib n = fibs !! n

-- Infinite lists with lazy evaluation
primes :: [Int]
primes = 2 : [x | x <- [3,5..], all (\p -> x `mod` p /= 0) (takeWhile (\p -> p*p <= x) primes)]

-- Sieve of Eratosthenes (lazy)
sieve :: [Int] -> [Int]
sieve (p:xs) = p : sieve [x | x <- xs, x `mod` p /= 0]
primesSieve :: [Int]
primesSieve = sieve [2..]
```

### Best Practices for Performance

**1. Choose the Right Recursion Pattern:**

```haskell
-- Use linear recursion for simple, one-pass operations
map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs

-- Use tail recursion for operations that can be done incrementally
sum' :: [Int] -> Int
sum' xs = go xs 0
  where
    go [] acc = acc
    go (x:xs) acc = go xs (x + acc)

-- Use tree recursion only when necessary, and optimize with memoization
fibTree :: Int -> Int
fibTree n = fibMemo n
  where
    fibMemo = (map fib [0..] !!)
    fib 0 = 0
    fib 1 = 1
    fib n = fibMemo (n-1) + fibMemo (n-2)
```

**2. Avoid Common Performance Pitfalls:**

```haskell
-- BAD: O(n²) due to repeated ++ operations
reverseBad :: [a] -> [a]
reverseBad [] = []
reverseBad (x:xs) = reverseBad xs ++ [x]

-- GOOD: O(n) using accumulator
reverseGood :: [a] -> [a]
reverseGood xs = go xs []
  where
    go [] acc = acc
    go (x:xs) acc = go xs (x:acc)

-- BAD: O(n²) due to repeated length calls
middleBad :: [a] -> Maybe a
middleBad [] = Nothing
middleBad xs = Just (xs !! (length xs `div` 2))

-- GOOD: O(n) using two pointers
middleGood :: [a] -> Maybe a
middleGood [] = Nothing
middleGood xs = Just (fst (go xs xs))
  where
    go (_:xs) (_:_:ys) = go xs ys
    go (x:_) _ = (x, [])
    go [] _ = error "Impossible"
```

**3. Profile and Measure:**

```haskell
-- Use profiling to identify bottlenecks
-- Compile with: ghc -prof -fprof-auto -rtsopts program.hs
-- Run with: ./program +RTS -p

-- Example of profiling-friendly code
expensiveOperation :: [Int] -> [Int]
expensiveOperation xs = 
    let result1 = map (* 2) xs
        result2 = filter even result1
        result3 = sort result2
    in result3

-- Use strict evaluation where appropriate
import Control.DeepSeq

strictSum :: [Int] -> Int
strictSum xs = go xs 0
  where
    go [] acc = acc
    go (x:xs) acc = 
        let newAcc = acc + x
        in newAcc `seq` go xs newAcc
```

## Research Papers and Further Reading

### Foundational Papers
1. **"Compiling Pattern Matching" (1987)** - Lennart Augustsson
   - Essential reading for understanding how pattern matching is compiled
   - Covers the compilation of pattern matching to efficient decision trees

2. **"The Implementation of Functional Programming Languages" (1987)** - Simon Peyton Jones
   - Comprehensive guide to implementing functional languages
   - Covers lazy evaluation, pattern matching, and recursion

3. **"Recursion Schemes for Dynamic Programming" (2004)** - Jeremy Gibbons
   - Advanced techniques for optimizing recursive algorithms
   - Covers catamorphisms, anamorphisms, and other recursion schemes

### Advanced Topics
1. **"Generic Programming with Folds and Unfolds" (1999)** - Jeremy Gibbons and Geraint Jones
   - Advanced patterns for working with recursive data types
   - Covers the theory behind fold and unfold operations

2. **"Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire" (1991)** - Erik Meijer
   - Classic paper on recursion schemes
   - Introduces the "banana" notation for catamorphisms

3. **"A Tutorial on the Universality and Expressiveness of Fold" (1999)** - Graham Hutton
   - Deep dive into the power and expressiveness of fold operations
   - Shows how fold can express many common programming patterns

### Performance and Optimization
1. **"Strictness Analysis" (1987)** - Alan Mycroft
   - Techniques for determining when strict evaluation is beneficial
   - Important for optimizing Haskell programs

2. **"Lazy Functional State Threads" (1994)** - John Launchbury and Simon Peyton Jones
   - How to handle state in functional programs efficiently
   - Covers the ST monad and related techniques

## Conclusion

Pattern matching and recursion are not just programming techniques—they are fundamental ways of thinking about computation. They enable us to write code that is:

- **Declarative**: We describe what we want, not how to get it
- **Composable**: Small functions combine to solve complex problems
- **Correct**: Pattern matching ensures we handle all cases
- **Efficient**: With proper optimization, recursive solutions can be very fast

Mastering these concepts opens the door to elegant, maintainable, and powerful functional programming. The key is to:

1. **Start simple**: Begin with basic pattern matching and linear recursion
2. **Understand the patterns**: Learn when to use each type of recursion
3. **Optimize carefully**: Use profiling to identify bottlenecks
4. **Practice regularly**: Work through problems using these techniques
5. **Read the literature**: Study the foundational papers and advanced techniques

Remember: the goal is not just to write code that works, but to write code that is beautiful, efficient, and maintainable. Pattern matching and recursion are your tools for achieving this goal.