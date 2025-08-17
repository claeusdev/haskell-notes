# Section 2: Basic Syntax and Types

This section provides a comprehensive foundation in Haskell's syntax and type system, which forms the core of all Haskell programming.

## 📁 Files in this Section

- **`notes.md`** - Detailed explanation of Haskell's type system, syntax, and fundamental concepts
- **`examples.md`** - Practical code examples demonstrating every concept with runnable code
- **`Calculator.hs`** - A complete calculator project showcasing data types, pattern matching, and error handling

## 🎯 Learning Objectives

After completing this section, you should master:

1. **Type System Fundamentals**
   - Primitive types (Int, Double, Bool, Char, String)
   - Type signatures and type inference
   - Polymorphic types and type variables
   - Type constraints and type classes

2. **Data Structures**
   - Lists: construction, operations, comprehensions, ranges
   - Tuples: creation, pattern matching, when to use vs lists
   - Strings as lists of characters

3. **Function Syntax**
   - Function definition and application
   - Currying and partial application
   - Infix vs prefix notation
   - Lambda functions (anonymous functions)

4. **Pattern Matching**
   - Basic patterns (literals, variables, wildcards)
   - List patterns (empty list, cons, multiple elements)
   - Tuple patterns and deconstruction
   - Guards for conditional logic
   - Where clauses for local definitions

5. **Error Handling**
   - Maybe type for safe operations
   - Either type for detailed error information
   - Safe alternatives to partial functions

## 🛠 Key Concepts Demonstrated

### Type System Strength
```haskell
-- Type safety prevents runtime errors
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing  -- Compile-time guarantee of safety
safeDivide x y = Just (x / y)
```

### Pattern Matching Power
```haskell
-- Elegant data deconstruction
processResult :: Maybe Int -> String
processResult Nothing = "No result"
processResult (Just n) = "Result: " ++ show n
```

### Functional Composition
```haskell
-- Function composition and higher-order functions
processNumbers :: [Int] -> [Int]
processNumbers = map (*2) . filter (>0)
```

## 🎮 Project: Advanced Calculator

The `Calculator.hs` project demonstrates:

### Core Features
- **Multiple Data Types**: Operations, Expressions, Results, Calculator State
- **Comprehensive Error Handling**: Division by zero, negative square roots, etc.
- **Pattern Matching**: Extensive use throughout expression evaluation
- **Parsing**: String to expression conversion
- **State Management**: Memory operations and calculation history
- **Interactive REPL**: Command-line interface with help system

### Advanced Features
- **Unary Operations**: sqrt, abs, negation
- **Binary Operations**: +, -, *, /, ^, %
- **Special Functions**: factorial, GCD, LCM, statistical operations
- **Number Base Conversion**: decimal to binary/hex
- **Memory System**: store, recall, clear operations
- **History Tracking**: last 10 calculations

### To Run the Calculator
```bash
# Compile and run
ghc Calculator.hs -o calculator
./calculator

# Or run directly
runhaskell Calculator.hs

# Or in GHCi
ghci Calculator.hs
*Main> main
```

### Example Calculator Session
```
Calculator> 5 + 3
8
Calculator> sqrt 16
4
Calculator> 10 / 0
Error: Division by zero
Calculator> help
[Shows help menu]
Calculator> quit
```

## 📚 Research Papers

### Essential Type System Papers
1. **"Principal Type-Schemes for Functional Programs" (1982)**
   - Authors: Luis Damas and Robin Milner
   - [Paper Link](https://web.cs.wpi.edu/~cs4536/c12/milner-damas_principal_types.pdf)
   - Foundation of Haskell's type inference

2. **"A Theory of Type Polymorphism in Programming" (1978)**
   - Author: Robin Milner
   - Introduces the Hindley-Milner type system

3. **"How to Make Ad-hoc Polymorphism Less Ad Hoc" (1989)**
   - Authors: Philip Wadler and Stephen Blott
   - [Paper Link](https://people.csail.mit.edu/dnj/teaching/6898/papers/wadler88.pdf)
   - Introduction of type classes

## 💡 Practice Exercises

### Beginner
1. Create functions for basic geometric calculations (area, perimeter)
2. Implement safe list operations (safeHead, safeTail, safeIndex)
3. Write functions using guards for different conditional logic
4. Practice list comprehensions for data filtering and transformation

### Intermediate
1. Extend the calculator with trigonometric functions
2. Implement a simple expression parser for parentheses
3. Create a grade calculator with letter grade conversion
4. Build a text processing tool with word/character statistics

### Advanced
1. Add complex number support to the calculator
2. Implement a simple programming language evaluator
3. Create a type-safe configuration parser
4. Build a mathematical expression simplifier

## 🔧 Development Tips

### Type-Driven Development
1. Start with type signatures
2. Let the compiler guide implementation
3. Use holes (`_`) during development
4. Leverage type inference for rapid prototyping

### Debugging Type Errors
```bash
# In GHCi
:type expression     -- Check expression type
:info TypeClass      -- Get type class information
:kind Type           -- Check type's kind
```

### Best Practices
- Always provide type signatures for top-level functions
- Use pattern matching instead of conditional logic when possible
- Prefer Maybe/Either over exceptions
- Keep functions small and focused
- Use meaningful variable names

## 🔗 Prerequisites

Before starting this section:
- Complete Section 1: Introduction to Haskell
- Have GHC and GHCi installed
- Basic understanding of functional programming concepts

## 🔗 Next Steps

After mastering this section:
1. **Section 3: Functions and Higher-Order Functions** - Deep dive into functional programming
2. **Section 4: Pattern Matching and Recursion** - Advanced pattern matching techniques
3. **Section 5: Algebraic Data Types** - Custom data type creation

## 📖 Additional Resources

### Books
- "Programming in Haskell" by Graham Hutton - Chapters 2-4
- "Learn You a Haskell for Great Good!" - Chapters 2-5
- "Real World Haskell" - Chapters 2-3

### Online Resources
- [Haskell Type System Tutorial](https://www.haskell.org/tutorial/types.html)
- [School of Haskell - Types](https://www.schoolofhaskell.com/school/starting-with-haskell/basics-of-haskell/3-types-and-type-classes)
- [Learn Haskell - Pattern Matching](https://learnyouahaskell.com/syntax-in-functions)

### Tools
- **GHCi** - Interactive development and testing
- **Hoogle** - Type-based function search
- **Hlint** - Haskell code suggestions
- **Haddock** - Documentation generation

Happy coding! The type system is your friend - embrace its power! 🚀