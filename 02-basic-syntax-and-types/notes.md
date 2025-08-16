# Basic Syntax and Types - Notes

## Overview

Haskell's type system is one of its most powerful features, providing strong compile-time guarantees while enabling elegant and expressive code. This section covers the fundamental syntax and type system concepts that form the foundation of Haskell programming.

## Primitive Types

### Numeric Types

#### Integer Types
- **`Int`**: Fixed-precision signed integers (typically 32 or 64 bits)
- **`Integer`**: Arbitrary-precision signed integers
- **`Word`**: Fixed-precision unsigned integers

#### Floating Point Types
- **`Float`**: Single-precision floating point
- **`Double`**: Double-precision floating point
- **`Rational`**: Arbitrary-precision rational numbers

### Character and String Types
- **`Char`**: Single Unicode character
- **`String`**: Type alias for `[Char]` (list of characters)
- **`Text`**: Efficient Unicode text (from `Data.Text`)
- **`ByteString`**: Efficient byte strings (from `Data.ByteString`)

### Boolean Type
- **`Bool`**: Boolean values (`True` or `False`)

### Unit Type
- **`()`**: Unit type with single value `()`, used for side effects

## Type Signatures

### Function Type Signatures
```haskell
-- Basic function type
functionName :: Type1 -> Type2 -> ReturnType

-- Example
add :: Int -> Int -> Int
add x y = x + y
```

### Polymorphic Types
```haskell
-- Type variables (lowercase letters)
identity :: a -> a
identity x = x

-- Multiple type variables
pair :: a -> b -> (a, b)
pair x y = (x, y)
```

### Type Constraints
```haskell
-- Single constraint
compare :: Ord a => a -> a -> Ordering

-- Multiple constraints
showAndCompare :: (Show a, Ord a) => a -> a -> String
```

## Type Inference

### How Type Inference Works
1. **Hindley-Milner Algorithm**: Automatically deduces types
2. **Principal Types**: Most general type possible
3. **Unification**: Resolves type variables
4. **Constraint Solving**: Handles type class constraints

### Benefits of Type Inference
- Reduces verbosity
- Catches type errors early
- Maintains type safety
- Enables rapid prototyping

### When to Add Type Signatures
- Top-level functions (best practice)
- Complex expressions
- Error disambiguation
- Documentation purposes

## Lists

### List Syntax
```haskell
-- Empty list
empty :: [a]
empty = []

-- List construction
numbers :: [Int]
numbers = [1, 2, 3, 4, 5]

-- Using cons operator
constructed :: [Int]
constructed = 1 : 2 : 3 : []
```

### List Operations
- **Head**: First element `head [1,2,3]` → `1`
- **Tail**: All but first `tail [1,2,3]` → `[2,3]`
- **Init**: All but last `init [1,2,3]` → `[1,2]`
- **Last**: Last element `last [1,2,3]` → `3`
- **Length**: Number of elements `length [1,2,3]` → `3`
- **Null**: Check if empty `null []` → `True`

### List Comprehensions
```haskell
-- Basic comprehension
squares = [x^2 | x <- [1..10]]

-- With predicate
evenSquares = [x^2 | x <- [1..10], even x]

-- Multiple generators
pairs = [(x, y) | x <- [1..3], y <- ['a'..'c']]
```

### Ranges
```haskell
-- Simple ranges
oneToTen = [1..10]
letters = ['a'..'z']

-- With step
evens = [2, 4..20]
countdown = [10, 9..1]

-- Infinite lists
naturals = [1..]
powers = [2^n | n <- [0..]]
```

## Tuples

### Tuple Types
```haskell
-- Pair
point :: (Int, Int)
point = (3, 4)

-- Triple
person :: (String, Int, Bool)
person = ("Alice", 30, True)

-- Heterogeneous
mixed :: (String, [Int], Char)
mixed = ("hello", [1,2,3], 'x')
```

### Tuple Operations
```haskell
-- Pattern matching
getName :: (String, Int) -> String
getName (name, _) = name

-- Built-in functions (pairs only)
first = fst (1, 2)    -- 1
second = snd (1, 2)   -- 2
```

### When to Use Tuples vs Lists
- **Tuples**: Fixed number of heterogeneous elements
- **Lists**: Variable number of homogeneous elements

## Function Syntax

### Basic Function Definition
```haskell
-- Single parameter
square :: Int -> Int
square x = x * x

-- Multiple parameters
add :: Int -> Int -> Int
add x y = x + y

-- No parameters (constant)
pi :: Double
pi = 3.14159
```

### Currying and Partial Application
```haskell
-- All functions are curried
add :: Int -> Int -> Int
add x y = x + y

-- Partial application
addFive :: Int -> Int
addFive = add 5

-- Equivalent to
addFive' :: Int -> Int
addFive' y = add 5 y
```

### Infix Functions
```haskell
-- Infix operators
result = 3 + 4
result' = (+) 3 4  -- Prefix form

-- Making functions infix
add x y = x + y
result'' = 3 `add` 4
```

### Lambda Functions
```haskell
-- Anonymous function
square = \x -> x * x

-- Multiple parameters
add = \x y -> x + y

-- In higher-order contexts
doubled = map (\x -> x * 2) [1,2,3,4]
```

## Pattern Matching

### Basic Patterns
```haskell
-- Literal patterns
isZero :: Int -> Bool
isZero 0 = True
isZero _ = False

-- Variable patterns
identity :: a -> a
identity x = x

-- Wildcard patterns
ignore :: (a, b) -> a
ignore (x, _) = x
```

### List Patterns
```haskell
-- Empty list
isEmpty :: [a] -> Bool
isEmpty [] = True
isEmpty _ = False

-- Cons pattern
head' :: [a] -> a
head' (x:_) = x

-- Multiple elements
firstTwo :: [a] -> (a, a)
firstTwo (x:y:_) = (x, y)
```

### Tuple Patterns
```haskell
-- Pair decomposition
swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)

-- Triple patterns
getFirst :: (a, b, c) -> a
getFirst (x, _, _) = x
```

### Guards
```haskell
-- Guard syntax
absoluteValue :: Int -> Int
absoluteValue x
    | x >= 0 = x
    | otherwise = -x

-- Multiple guards
grade :: Int -> Char
grade score
    | score >= 90 = 'A'
    | score >= 80 = 'B'
    | score >= 70 = 'C'
    | score >= 60 = 'D'
    | otherwise = 'F'
```

### Where Clauses
```haskell
-- Local definitions
bmiCategory :: Float -> Float -> String
bmiCategory weight height
    | bmi < 18.5 = "Underweight"
    | bmi < 25.0 = "Normal"
    | bmi < 30.0 = "Overweight"
    | otherwise = "Obese"
    where bmi = weight / height^2
```

### Let Expressions
```haskell
-- Local bindings in expressions
cylinder :: Float -> Float -> Float
cylinder r h = 
    let sideArea = 2 * pi * r * h
        topArea = pi * r^2
    in sideArea + 2 * topArea
```

## Type Classes Introduction

### What are Type Classes?
Type classes define sets of functions that can be implemented by different types, providing ad-hoc polymorphism.

### Common Type Classes

#### Eq (Equality)
```haskell
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
```

#### Ord (Ordering)
```haskell
class Eq a => Ord a where
    compare :: a -> a -> Ordering
    (<), (<=), (>), (>=) :: a -> a -> Bool
```

#### Show (String Representation)
```haskell
class Show a where
    show :: a -> String
```

#### Read (Parsing from String)
```haskell
class Read a where
    read :: String -> a
```

#### Num (Numeric)
```haskell
class Num a where
    (+), (-), (*) :: a -> a -> a
    negate :: a -> a
    abs :: a -> a
    signum :: a -> a
    fromInteger :: Integer -> a
```

## Error Handling

### Maybe Type
```haskell
data Maybe a = Nothing | Just a

-- Safe operations
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x
```

### Either Type
```haskell
data Either a b = Left a | Right b

-- Error with message
parseNumber :: String -> Either String Int
parseNumber s = case reads s of
    [(n, "")] -> Right n
    _ -> Left ("Invalid number: " ++ s)
```

## Syntax Sugar

### Do Notation (Preview)
```haskell
-- For sequential operations
main :: IO ()
main = do
    putStrLn "Enter your name:"
    name <- getLine
    putStrLn ("Hello, " ++ name)
```

### Record Syntax
```haskell
-- Data type with named fields
data Person = Person
    { name :: String
    , age :: Int
    , email :: String
    }

-- Construction
john = Person "John" 30 "john@example.com"

-- Access
johnName = name john

-- Update
olderJohn = john { age = 31 }
```

## Research Papers

### Type System Foundations
1. **"Principal Type-Schemes for Functional Programs" (1982)**
   - Authors: Luis Damas and Robin Milner
   - [Link](https://web.cs.wpi.edu/~cs4536/c12/milner-damas_principal_types.pdf)
   - Foundation of Haskell's type inference algorithm

2. **"A Theory of Type Polymorphism in Programming" (1978)**
   - Author: Robin Milner
   - Introduces the Hindley-Milner type system

### Type Classes
1. **"How to Make Ad-hoc Polymorphism Less Ad Hoc" (1989)**
   - Authors: Philip Wadler and Stephen Blott
   - [Link](https://people.csail.mit.edu/dnj/teaching/6898/papers/wadler88.pdf)
   - Introduction of type classes to Haskell

2. **"Type Classes in Haskell" (1996)**
   - Authors: Cordelia Hall, Kevin Hammond, Simon Peyton Jones, Philip Wadler
   - Comprehensive overview of type class design

### List Processing
1. **"Why Functional Programming Matters" (1990)**
   - Author: John Hughes
   - Demonstrates power of list processing and lazy evaluation

## Best Practices

### Type Signatures
- Always provide type signatures for top-level functions
- Use type signatures for documentation
- Prefer explicit over implicit when unclear

### Naming Conventions
- Use camelCase for functions and variables
- Use PascalCase for types and constructors
- Use descriptive names over abbreviations

### Pattern Matching
- Handle all cases or use wildcards
- Order patterns from specific to general
- Use guards for complex conditions

### Error Handling
- Use Maybe for operations that might fail
- Use Either for detailed error information
- Avoid partial functions in production code

### Code Organization
- Group related functions together
- Use appropriate module structure
- Separate pure and impure code

## Common Pitfalls

### Type Errors
- Missing type class constraints
- Infinite types from recursive definitions
- Ambiguous type variables

### Performance Issues
- Excessive list concatenation (++)
- Using head/tail on empty lists
- Not being aware of laziness implications

### Syntax Confusion
- Mixing up (,) and [] for tuples vs lists
- Forgetting parentheses in complex expressions
- Incorrect indentation in do blocks

## Tools and Debugging

### GHCi Commands
- `:type expr` - Show type of expression
- `:info name` - Show information about name
- `:kind type` - Show kind of type
- `:browse Module` - Show module contents

### Type Error Interpretation
- Read from inside out
- Look for type mismatches
- Check function arity
- Verify type class constraints

### Development Workflow
1. Write type signatures first
2. Implement with holes (`_`)
3. Use GHCi for testing
4. Refactor with confidence

This foundation in Haskell's syntax and type system is crucial for all advanced topics. The strong static typing provides safety guarantees while the inference system maintains expressiveness.