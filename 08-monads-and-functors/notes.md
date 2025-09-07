# Monads and Functors - Comprehensive Notes

## Overview

Monads and Functors are abstract patterns that capture common computation structures. They provide a mathematical foundation for handling effects, sequencing operations, and composing computations in a pure functional language. Understanding these abstractions is crucial for writing elegant, composable, and maintainable Haskell code.

**Key Learning Objectives:**
- Master the Functor type class and its laws
- Understand Applicative functors and their applications
- Learn the Monad type class and monadic patterns
- Explore common monads and their use cases
- Understand monad transformers for combining effects
- Apply monadic patterns to solve real-world problems

## Functors: Mapping Over Contexts

### The Functor Type Class: The Foundation

A Functor is a type class that represents types that can be mapped over. It provides a way to apply functions to values inside a context without changing the context itself.

**Functor Definition and Laws:**
```haskell
class Functor f where
    fmap :: (a -> b) -> f a -> f b
    
    -- Infix operator
    (<$>) :: (a -> b) -> f a -> f b
    (<$>) = fmap

-- Functor Laws (must be satisfied by all instances):
-- 1. Identity: fmap id = id
-- 2. Composition: fmap (f . g) = fmap f . fmap g
```

**Understanding Functors:**
- **Context Preservation**: The structure/context is preserved
- **Function Application**: Functions are applied to values inside the context
- **Composability**: Functors compose naturally
- **Laws**: Ensure predictable behavior

### Common Functors: Building Blocks

**Maybe Functor:**
```haskell
-- Maybe represents optional values
instance Functor Maybe where
    fmap _ Nothing = Nothing
    fmap f (Just x) = Just (f x)

-- Usage examples
safeIncrement :: Maybe Int -> Maybe Int
safeIncrement = fmap (+1)

-- fmap (+1) Nothing = Nothing
-- fmap (+1) (Just 5) = Just 6

-- Chaining operations
safeDoubleIncrement :: Maybe Int -> Maybe Int
safeDoubleIncrement = fmap (+1) . fmap (+1)
-- or equivalently: fmap ((+1) . (+1))
```

**List Functor:**
```haskell
-- Lists represent collections
instance Functor [] where
    fmap = map

-- Usage examples
doubleAll :: [Int] -> [Int]
doubleAll = fmap (*2)

-- fmap (*2) [1,2,3] = [2,4,6]

-- Chaining operations
processNumbers :: [Int] -> [Int]
processNumbers = fmap (*2) . fmap (+1)
-- or equivalently: fmap ((*2) . (+1))
```

**IO Functor:**
```haskell
-- IO represents computations with side effects
instance Functor IO where
    fmap f action = do
        x <- action
        return (f x)

-- Usage examples
readAndProcess :: FilePath -> IO String
readAndProcess path = fmap (map toUpper) (readFile path)

-- Chaining IO operations
readAndProcessFile :: FilePath -> IO [String]
readAndProcessFile path = fmap lines (readFile path)
```

**Either Functor:**
```haskell
-- Either represents computations that can fail
instance Functor (Either e) where
    fmap _ (Left e) = Left e
    fmap f (Right x) = Right (f x)

-- Usage examples
safeParse :: String -> Either String Int
safeParse s = case reads s of
    [(n, "")] -> Right n
    _ -> Left ("Invalid number: " ++ s)

safeIncrementEither :: Either String Int -> Either String Int
safeIncrementEither = fmap (+1)
```

## Applicative Functors

### The Applicative Type Class
```haskell
class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

-- Applicative Laws:
-- 1. pure id <*> v = v
-- 2. pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
-- 3. pure f <*> pure x = pure (f x)
-- 4. u <*> pure y = pure ($ y) <*> u
```

### Applicative Examples
```haskell
-- Maybe Applicative
instance Applicative Maybe where
    pure = Just
    Nothing <*> _ = Nothing
    (Just f) <*> something = fmap f something

-- Multi-argument functions
liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
liftA2 f x y = f <$> x <*> y

-- Validation with Either
validatePerson :: String -> Int -> String -> Either [String] Person
validatePerson name age email = Person 
    <$> validateName name 
    <*> validateAge age 
    <*> validateEmail email
```

## Monads

### The Monad Type Class
```haskell
class Applicative m => Monad m where
    (>>=) :: m a -> (a -> m b) -> m b
    return :: a -> m a
    return = pure

-- Monad Laws:
-- 1. return a >>= k = k a (left identity)
-- 2. m >>= return = m (right identity)  
-- 3. m >>= (\x -> k x >>= h) = (m >>= k) >>= h (associativity)
```

### Common Monads

#### Maybe Monad
```haskell
instance Monad Maybe where
    Nothing >>= _ = Nothing
    (Just x) >>= f = f x

-- Safe computation chains
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

computation :: Double -> Maybe Double
computation x = do
    y <- safeDivide x 2
    z <- safeDivide y 3
    safeDivide z 4
```

#### Either Monad
```haskell
instance Monad (Either e) where
    Left e >>= _ = Left e
    Right x >>= f = f x

-- Error handling with context
parseNumber :: String -> Either String Int
parseNumber s = case reads s of
    [(n, "")] -> Right n
    _ -> Left ("Invalid number: " ++ s)

calculate :: String -> String -> Either String Int
calculate xs ys = do
    x <- parseNumber xs
    y <- parseNumber ys
    if y == 0 
        then Left "Division by zero"
        else Right (x `div` y)
```

#### State Monad
```haskell
newtype State s a = State { runState :: s -> (a, s) }

instance Monad (State s) where
    return x = State $ \s -> (x, s)
    m >>= k = State $ \s -> 
        let (a, s') = runState m s
            (b, s'') = runState (k a) s'
        in (b, s'')

-- State operations
get :: State s s
get = State $ \s -> (s, s)

put :: s -> State s ()
put s = State $ \_ -> ((), s)

modify :: (s -> s) -> State s ()
modify f = State $ \s -> ((), f s)
```

#### IO Monad
```haskell
-- IO operations
main :: IO ()
main = do
    putStrLn "Enter your name:"
    name <- getLine
    putStrLn ("Hello, " ++ name)
    
-- File operations
readAndProcess :: FilePath -> IO String
readAndProcess filename = do
    content <- readFile filename
    return (map toUpper content)
```

## Monad Transformers

### The MonadTrans Class
```haskell
class MonadTrans t where
    lift :: Monad m => m a -> t m a

-- Example: ReaderT transformer
newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }

instance MonadTrans (ReaderT r) where
    lift m = ReaderT $ \_ -> m
```

### Common Transformers
```haskell
-- StateT - State + other monad
type StateT s m a = StateT { runStateT :: s -> m (a, s) }

-- ReaderT - Reader + other monad  
type ReaderT r m a = ReaderT { runReaderT :: r -> m a }

-- ExceptT - Either + other monad
type ExceptT e m a = ExceptT { runExceptT :: m (Either e a) }
```

## do Notation

### Desugaring do
```haskell
-- This do block:
computation = do
    x <- action1
    y <- action2 x
    action3 x y

-- Desugars to:
computation = action1 >>= \x ->
              action2 x >>= \y ->
              action3 x y
```

### do with Different Monads
```haskell
-- Maybe monad
maybeComputation :: Maybe Int
maybeComputation = do
    x <- Just 5
    y <- Just 3
    return (x + y)

-- List monad
listComputation :: [Int]
listComputation = do
    x <- [1, 2, 3]
    y <- [10, 20]
    return (x + y)
```

## Research Papers

### Foundational Papers
1. **"Monads for Functional Programming" (1995)** - Philip Wadler
   - [Link](https://homepages.inf.ed.ac.uk/wadler/papers/marktoberdorf/baastad.pdf)
2. **"The Essence of Functional Programming" (1992)** - Philip Wadler
3. **"Applicative Programming with Effects" (2008)** - Conor McBride and Ross Paterson

### Advanced Topics
1. **"Monad Transformers and Modular Interpreters" (1995)** - Sheng Liang, Paul Hudak, Mark Jones
2. **"Extensible Effects" (2013)** - Oleg Kiselyov, Amr Sabry, Cameron Swords

## Common Patterns

### Sequencing Operations
```haskell
-- Sequence IO actions
sequence_ :: Monad m => [m a] -> m ()
sequence_ = foldr (>>) (return ())

-- Map and sequence
mapM :: Monad m => (a -> m b) -> [a] -> m [b]
mapM f xs = sequence (map f xs)
```

### Error Handling
```haskell
-- Maybe chain
safeOperation :: String -> Maybe Int
safeOperation input = do
    n <- readMay input
    guard (n > 0)
    return (n * 2)

-- Either chain with errors
validateInput :: String -> Either String Int
validateInput input = do
    n <- parseNumber input
    when (n < 0) $ Left "Number must be positive"
    return n
```

Monads provide a unified way to handle effects while maintaining purity and composability in Haskell programs.