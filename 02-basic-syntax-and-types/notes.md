# Basic Syntax and Types - Comprehensive Notes

## Overview

Haskell's type system is one of its most powerful features, providing strong compile-time guarantees while enabling elegant and expressive code. This comprehensive guide covers the fundamental syntax and type system concepts that form the foundation of Haskell programming, from basic types to advanced type system features.

**Key Learning Objectives:**
- Master Haskell's syntax and type system fundamentals
- Understand primitive types and their characteristics
- Learn type inference and when to use explicit type signatures
- Explore pattern matching and its power
- Understand type classes and their role in polymorphism
- Learn best practices for type-safe programming

## Primitive Types: The Building Blocks

### Numeric Types: Precision and Performance

Haskell provides several numeric types, each optimized for different use cases and precision requirements.

#### Integer Types: Whole Numbers

**`Int` - Fixed-Precision Signed Integers**
- **Size**: Typically 32 or 64 bits (platform-dependent)
- **Range**: -2^63 to 2^63-1 (64-bit) or -2^31 to 2^31-1 (32-bit)
- **Performance**: Fastest integer operations
- **Use cases**: Counters, indices, small calculations

```haskell
-- Int examples
maxInt :: Int
maxInt = maxBound :: Int

minInt :: Int
minInt = minBound :: Int

-- Int arithmetic
addInts :: Int -> Int -> Int
addInts x y = x + y

-- Int conversion
intToInteger :: Int -> Integer
intToInteger = toInteger
```

**`Integer` - Arbitrary-Precision Signed Integers**
- **Size**: Unlimited precision (limited only by available memory)
- **Range**: No practical limit
- **Performance**: Slower than Int for small numbers
- **Use cases**: Cryptography, large calculations, exact arithmetic

```haskell
-- Integer examples
largeNumber :: Integer
largeNumber = 2^1000  -- No overflow!

-- Factorial with Integer (no overflow)
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Large calculations
fibonacci :: Integer -> Integer
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

-- Can handle very large numbers
hugeFib = fibonacci 1000
```

**`Word` - Fixed-Precision Unsigned Integers**
- **Size**: Same as Int (32 or 64 bits)
- **Range**: 0 to 2^64-1 (64-bit) or 0 to 2^32-1 (32-bit)
- **Performance**: Same as Int
- **Use cases**: Bit manipulation, array indices, counters that can't be negative

```haskell
-- Word examples
maxWord :: Word
maxWord = maxBound :: Word

-- Word arithmetic
addWords :: Word -> Word -> Word
addWords x y = x + y

-- Word to Int conversion (may overflow)
wordToInt :: Word -> Int
wordToInt = fromIntegral
```

#### Floating Point Types: Decimal Numbers

**`Float` - Single-Precision Floating Point**
- **Size**: 32 bits
- **Precision**: ~7 decimal digits
- **Performance**: Faster than Double
- **Use cases**: Graphics, embedded systems, when memory is limited

```haskell
-- Float examples
piFloat :: Float
piFloat = 3.1415927

-- Float arithmetic
circleArea :: Float -> Float
circleArea r = piFloat * r * r

-- Float precision issues
precisionExample :: Float
precisionExample = 0.1 + 0.2  -- May not equal 0.3 exactly
```

**`Double` - Double-Precision Floating Point**
- **Size**: 64 bits
- **Precision**: ~15 decimal digits
- **Performance**: Slower than Float but more precise
- **Use cases**: Scientific computing, financial calculations, general-purpose floating point

```haskell
-- Double examples
piDouble :: Double
piDouble = 3.141592653589793

-- Double arithmetic
circleAreaDouble :: Double -> Double
circleAreaDouble r = pi * r * r

-- Better precision
precisionExampleDouble :: Double
precisionExampleDouble = 0.1 + 0.2  -- Closer to 0.3
```

**`Rational` - Arbitrary-Precision Rational Numbers**
- **Size**: Unlimited precision
- **Precision**: Exact (no rounding errors)
- **Performance**: Slower than Float/Double
- **Use cases**: Exact arithmetic, financial calculations, symbolic computation

```haskell
-- Rational examples
import Data.Ratio

-- Rational literals
oneThird :: Rational
oneThird = 1 % 3

-- Exact arithmetic
exactSum :: Rational
exactSum = 1 % 3 + 1 % 3  -- Exactly 2 % 3

-- Rational to Double conversion
rationalToDouble :: Rational -> Double
rationalToDouble r = fromRational r
```

### Character and String Types: Text Processing

**`Char` - Single Unicode Character**
- **Size**: 32 bits (Unicode code point)
- **Range**: 0 to 0x10FFFF (Unicode range)
- **Performance**: Very fast
- **Use cases**: Single character processing, pattern matching

```haskell
-- Char examples
-- Character literals
newline :: Char
newline = '\n'

tab :: Char
tab = '\t'

-- Unicode characters
euro :: Char
euro = '€'

-- Char functions
isDigit :: Char -> Bool
isDigit c = c >= '0' && c <= '9'

isAlpha :: Char -> Bool
isAlpha c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')

-- Char to Int conversion
charToInt :: Char -> Int
charToInt c = fromEnum c
```

**`String` - List of Characters**
- **Type**: `type String = [Char]`
- **Performance**: Slow for large text (linked list)
- **Use cases**: Small strings, learning, simple text processing

```haskell
-- String examples
-- String literals
greeting :: String
greeting = "Hello, World!"

-- String functions
stringLength :: String -> Int
stringLength = length

-- String concatenation
concatenate :: String -> String -> String
concatenate s1 s2 = s1 ++ s2

-- String processing
words :: String -> [String]
words = Data.List.words

unwords :: [String] -> String
unwords = Data.List.unwords
```

**`Text` - Efficient Unicode Text**
- **Type**: From `Data.Text` package
- **Performance**: Much faster than String for large text
- **Memory**: More memory efficient than String
- **Use cases**: Production text processing, large documents

```haskell
-- Text examples
import Data.Text as T

-- Text literals
greetingText :: Text
greetingText = "Hello, World!"

-- Text functions
textLength :: Text -> Int
textLength = T.length

-- Text concatenation
concatenateText :: Text -> Text -> Text
concatenateText t1 t2 = t1 <> t2

-- Text processing
wordsText :: Text -> [Text]
wordsText = T.words

-- String to Text conversion
stringToText :: String -> Text
stringToText = T.pack

-- Text to String conversion
textToString :: Text -> String
textToString = T.unpack
```

**`ByteString` - Efficient Byte Sequences**
- **Type**: From `Data.ByteString` package
- **Performance**: Very fast for binary data
- **Memory**: Most memory efficient for binary data
- **Use cases**: Binary file I/O, network protocols, performance-critical text processing

```haskell
-- ByteString examples
import Data.ByteString as B
import Data.ByteString.Char8 as BC

-- ByteString literals
helloBytes :: ByteString
helloBytes = BC.pack "Hello"

-- ByteString functions
byteStringLength :: ByteString -> Int
byteStringLength = B.length

-- ByteString concatenation
concatenateBytes :: ByteString -> ByteString -> ByteString
concatenateBytes b1 b2 = b1 <> b2

-- String to ByteString conversion
stringToByteString :: String -> ByteString
stringToByteString = BC.pack
```

### Boolean Type: Logical Values

**`Bool` - Boolean Values**
- **Values**: `True` and `False`
- **Size**: 1 bit (but typically stored as 1 byte)
- **Performance**: Very fast
- **Use cases**: Conditional logic, predicates, flags

```haskell
-- Bool examples
-- Boolean literals
alwaysTrue :: Bool
alwaysTrue = True

alwaysFalse :: Bool
alwaysFalse = False

-- Boolean operations
and :: Bool -> Bool -> Bool
and True True = True
and _ _ = False

or :: Bool -> Bool -> Bool
or False False = False
or _ _ = True

not :: Bool -> Bool
not True = False
not False = True

-- Boolean functions
isEven :: Int -> Bool
isEven n = n `mod` 2 == 0

isPositive :: Int -> Bool
isPositive n = n > 0
```

### Unit Type: The Empty Type

**`()` - Unit Type**
- **Values**: Only `()`
- **Size**: 0 bits (optimized away)
- **Use cases**: Side effects, placeholder values, monadic operations

```haskell
-- Unit type examples
-- Unit literal
unitValue :: ()
unitValue = ()

-- Functions returning unit
printMessage :: String -> IO ()
printMessage msg = putStrLn msg

-- Unit in pattern matching
processUnit :: () -> String
processUnit () = "Got unit value"

-- Unit as placeholder
ignoreValue :: a -> ()
ignoreValue _ = ()
```

## Type Signatures: Communicating Intent

### Function Type Signatures: The Contract

Type signatures serve as contracts between functions and their callers, providing documentation and enabling type checking.

**Basic Function Types:**
```haskell
-- Simple function type
add :: Int -> Int -> Int
add x y = x + y

-- Function with no arguments (constant)
pi :: Double
pi = 3.141592653589793

-- Function with one argument
square :: Int -> Int
square x = x * x

-- Function with multiple arguments
distance :: Double -> Double -> Double -> Double -> Double
distance x1 y1 x2 y2 = sqrt ((x2 - x1)^2 + (y2 - y1)^2)
```

**Understanding Arrow Types:**
- `->` is right-associative
- `Int -> Int -> Int` means `Int -> (Int -> Int)`
- Functions are curried by default
- All functions take exactly one argument and return one value

```haskell
-- These are equivalent
add :: Int -> Int -> Int
add :: Int -> (Int -> Int)

-- Partial application
addFive :: Int -> Int
addFive = add 5

-- Function application
result :: Int
result = add 5 3  -- Same as (add 5) 3
```

### Polymorphic Types: One Function, Many Types

Polymorphic functions can work with multiple types, making code more reusable and expressive.

**Type Variables:**
```haskell
-- Single type variable
identity :: a -> a
identity x = x

-- Multiple type variables
pair :: a -> b -> (a, b)
pair x y = (x, y)

-- Type variables in different positions
swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)

-- Complex polymorphic function
map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs
```

**Constrained Polymorphism:**
```haskell
-- Single constraint
compare :: Ord a => a -> a -> Ordering
compare x y = if x < y then LT else if x > y then GT else EQ

-- Multiple constraints
showAndCompare :: (Show a, Ord a) => a -> a -> String
showAndCompare x y = 
    if x < y 
        then show x ++ " < " ++ show y
        else show x ++ " >= " ++ show y

-- Constraint with multiple type variables
zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith _ [] _ = []
zipWith _ _ [] = []
zipWith f (x:xs) (y:ys) = f x y : zipWith f xs ys
```

### Type Inference: Let the Compiler Work

Haskell's type inference system automatically deduces types, reducing verbosity while maintaining type safety.

**How Type Inference Works:**
1. **Hindley-Milner Algorithm**: Unifies type constraints
2. **Principal Types**: Finds the most general type
3. **Constraint Solving**: Resolves type class constraints
4. **Error Detection**: Catches type mismatches

```haskell
-- Type inference examples
-- Haskell infers: f :: a -> a
f x = x

-- Haskell infers: g :: Num a => a -> a
g x = x + 1

-- Haskell infers: h :: (Eq a, Num a) => a -> a -> Bool
h x y = x + y == 0

-- Haskell infers: i :: [a] -> [a]
i xs = xs

-- Haskell infers: j :: (a -> b) -> [a] -> [b]
j f xs = map f xs
```

**When to Add Explicit Type Signatures:**
- **Top-level functions**: Best practice for documentation
- **Complex expressions**: Help with debugging
- **Error disambiguation**: When compiler can't infer
- **Documentation**: Make intent clear

```haskell
-- Good: Explicit type signature
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = 
    quicksort (filter (< x) xs) ++ 
    [x] ++ 
    quicksort (filter (>= x) xs)

-- Good: Complex function with explicit signature
parseConfig :: String -> Either String Config
parseConfig input = 
    case parseConfigFile input of
        Left err -> Left ("Parse error: " ++ err)
        Right config -> Right config
```

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