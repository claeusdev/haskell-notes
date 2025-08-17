-- DataProcessor.hs
-- A comprehensive data processing tool demonstrating higher-order functions
-- This project covers: function composition, map/filter/fold, currying, pipelines

{-# LANGUAGE OverloadedStrings #-}

module DataProcessor where

import Data.List (sort, sortBy, group, groupBy, transpose, foldl')
import Data.Char (toUpper, toLower, isSpace, isAlpha, isDigit)
import Data.Function (on)
import Text.Read (readMaybe)

-- Data types for our processor
data Employee = Employee
    { empId :: Int
    , empName :: String
    , empDepartment :: String
    , empSalary :: Double
    , empYearsService :: Int
    } deriving (Show, Eq)

data ProcessingStats = ProcessingStats
    { totalRecords :: Int
    , validRecords :: Int
    , errorRecords :: Int
    , avgProcessingTime :: Double
    } deriving (Show)

-- CSV parsing and data transformation

-- Parse CSV line into Employee (simplified)
parseEmployee :: String -> Either String Employee
parseEmployee line = 
    case splitOn ',' line of
        [idStr, name, dept, salaryStr, yearsStr] -> do
            empId' <- parseField "ID" idStr readMaybe
            salary <- parseField "Salary" salaryStr readMaybe
            years <- parseField "Years" yearsStr readMaybe
            return $ Employee empId' (trim name) (trim dept) salary years
        _ -> Left "Invalid number of fields"
  where
    parseField fieldName str parser = 
        case parser (trim str) of
            Just val -> Right val
            Nothing -> Left $ "Invalid " ++ fieldName ++ ": " ++ str

-- Utility functions using higher-order patterns
splitOn :: Char -> String -> [String]
splitOn delimiter = foldr split [[]]
  where
    split char (word:words')
        | char == delimiter = []:word:words'
        | otherwise = (char:word):words'

trim :: String -> String
trim = reverse . dropWhile isSpace . reverse . dropWhile isSpace

-- Data analysis functions using higher-order functions

-- Calculate statistics using folds
calculateStats :: [Employee] -> (Double, Double, Double, Int)
calculateStats employees = (avgSalary, minSalary, maxSalary, totalEmployees)
  where
    salaries = map empSalary employees
    totalEmployees = length employees
    avgSalary = if totalEmployees > 0 
                then sum salaries / fromIntegral totalEmployees 
                else 0
    minSalary = if null salaries then 0 else minimum salaries
    maxSalary = if null salaries then 0 else maximum salaries

-- Group employees by department using groupBy
groupByDepartment :: [Employee] -> [(String, [Employee])]
groupByDepartment = map (\group' -> (empDepartment (head group'), group'))
                  . groupBy ((==) `on` empDepartment)
                  . sortBy (compare `on` empDepartment)

-- Filter functions with partial application
highEarners :: Double -> [Employee] -> [Employee]
highEarners threshold = filter ((>threshold) . empSalary)

seniorEmployees :: Int -> [Employee] -> [Employee]
seniorEmployees years = filter ((>=years) . empYearsService)

departmentFilter :: String -> [Employee] -> [Employee]
departmentFilter dept = filter ((==dept) . empDepartment)

-- Transformation pipelines using function composition
normalizeEmployee :: Employee -> Employee
normalizeEmployee emp = emp 
    { empName = normalizeName (empName emp)
    , empDepartment = normalizeDepartment (empDepartment emp)
    }
  where
    normalizeName = unwords . map capitalizeWord . words . map toLower
    normalizeDepartment = map toUpper
    capitalizeWord [] = []
    capitalizeWord (x:xs) = toUpper x : xs

-- Data processing pipeline
processEmployeeData :: [String] -> ([Employee], [String])
processEmployeeData = partitionEithers . map parseEmployee
  where
    partitionEithers = foldr partition ([], [])
    partition (Left err) (rights, lefts) = (rights, err:lefts)
    partition (Right val) (rights, lefts) = (val:rights, lefts)

-- Advanced analysis functions

-- Salary analysis by department
departmentSalaryAnalysis :: [Employee] -> [(String, (Double, Double, Int))]
departmentSalaryAnalysis = map analyzeDepartment . groupByDepartment
  where
    analyzeDepartment (dept, emps) = 
        let salaries = map empSalary emps
            avgSal = sum salaries / fromIntegral (length salaries)
            maxSal = maximum salaries
            count = length emps
        in (dept, (avgSal, maxSal, count))

-- Performance ranking
rankEmployees :: (Employee -> Double) -> [Employee] -> [(Int, Employee)]
rankEmployees metric = zip [1..] . sortBy (flip compare `on` metric)

-- Salary ranking
salaryRanking :: [Employee] -> [(Int, Employee)]
salaryRanking = rankEmployees empSalary

-- Experience ranking
experienceRanking :: [Employee] -> [(Int, Employee)]
experienceRanking = rankEmployees (fromIntegral . empYearsService)

-- Data validation using higher-order functions
validateEmployee :: Employee -> [String]
validateEmployee emp = 
    let validators = [ validateId (empId emp)
                     , validateName (empName emp)
                     , validateSalary (empSalary emp)
                     , validateYears (empYearsService emp)
                     ]
    in concat $ map (\f -> f emp) []  -- Apply validators
  where
    validateId id' emp
        | id' <= 0 = ["Invalid employee ID"]
        | otherwise = []
    validateName name emp
        | null (trim name) = ["Empty employee name"]
        | otherwise = []
    validateSalary salary emp
        | salary < 0 = ["Negative salary"]
        | otherwise = []
    validateYears years emp
        | years < 0 = ["Negative years of service"]
        | otherwise = []

-- Report generation using function composition and higher-order functions
generateReport :: [Employee] -> String
generateReport employees = unlines $ concat
    [ ["=== Employee Data Analysis Report ===", ""]
    , summarySection
    , [""]
    , departmentSection
    , [""]
    , topPerformersSection
    ]
  where
    (avgSal, minSal, maxSal, total) = calculateStats employees
    
    summarySection = 
        [ "SUMMARY:"
        , "Total Employees: " ++ show total
        , "Average Salary: $" ++ printf "%.2f" avgSal
        , "Salary Range: $" ++ printf "%.2f" minSal ++ " - $" ++ printf "%.2f" maxSal
        ]
    
    departmentSection = 
        "DEPARTMENT ANALYSIS:" : 
        map formatDeptStats (departmentSalaryAnalysis employees)
    
    topPerformersSection = 
        "TOP PERFORMERS BY SALARY:" :
        map formatRanking (take 5 $ salaryRanking employees)
    
    formatDeptStats (dept, (avg, max', count)) = 
        dept ++ ": " ++ show count ++ " employees, avg $" ++ 
        printf "%.2f" avg ++ ", max $" ++ printf "%.2f" max'
    
    formatRanking (rank, emp) = 
        show rank ++ ". " ++ empName emp ++ " (" ++ empDepartment emp ++ 
        ") - $" ++ printf "%.2f" (empSalary emp)

-- Simple printf implementation for formatting
printf :: String -> Double -> String
printf "%.2f" x = 
    let (whole, frac) = properFraction x
        fracStr = take 2 $ show (round (frac * 100)) ++ "00"
    in show whole ++ "." ++ fracStr

-- Data transformation utilities using map and fold
applyToColumn :: (a -> a) -> Int -> [[a]] -> [[a]]
applyToColumn f colIndex = map (\row -> 
    zipWith (\i val -> if i == colIndex then f val else val) [0..] row)

summarizeColumn :: (Num a) => Int -> [[a]] -> a
summarizeColumn colIndex matrix = sum $ map (!! colIndex) matrix

-- Functional data filtering and searching
searchEmployees :: (Employee -> Bool) -> [Employee] -> [Employee]
searchEmployees = filter

searchByName :: String -> [Employee] -> [Employee]
searchByName name = searchEmployees (isInfixOf (map toLower name) . map toLower . empName)
  where
    isInfixOf needle haystack = any (isPrefixOf needle) (tails haystack)
    isPrefixOf [] _ = True
    isPrefixOf _ [] = False
    isPrefixOf (x:xs) (y:ys) = x == y && isPrefixOf xs ys
    tails [] = [[]]
    tails xs@(_:ys) = xs : tails ys

-- Advanced function composition examples
processAndAnalyze :: [String] -> String
processAndAnalyze = generateReport           -- 4. Generate report
                  . map normalizeEmployee    -- 3. Normalize data
                  . fst                      -- 2. Extract valid employees
                  . processEmployeeData      -- 1. Parse CSV data

-- Complex data pipeline with error handling
safeProcessData :: [String] -> Either String String
safeProcessData csvLines = 
    case processEmployeeData csvLines of
        ([], errors) -> Left $ "No valid data found. Errors: " ++ unlines errors
        (employees, errors) -> 
            let report = generateReport (map normalizeEmployee employees)
                warningSection = if null errors 
                               then "" 
                               else "\nWARNINGS:\n" ++ unlines errors
            in Right (report ++ warningSection)

-- Interactive data analysis functions
analyzeByDepartment :: String -> [Employee] -> String
analyzeByDepartment dept employees = 
    let deptEmployees = departmentFilter dept employees
        (avg, min', max', count) = calculateStats deptEmployees
    in unlines
        [ "Department: " ++ dept
        , "Employees: " ++ show count
        , "Average Salary: $" ++ printf "%.2f" avg
        , "Salary Range: $" ++ printf "%.2f" min' ++ " - $" ++ printf "%.2f" max'
        ]

-- Sample data for testing
sampleData :: [String]
sampleData = 
    [ "1,John Doe,Engineering,75000,5"
    , "2,Jane Smith,Marketing,65000,3"
    , "3,Bob Johnson,Engineering,80000,7"
    , "4,Alice Brown,HR,55000,2"
    , "5,Charlie Wilson,Marketing,70000,4"
    , "6,Diana Lee,Engineering,85000,6"
    ]

-- Main demonstration function
main :: IO ()
main = do
    putStrLn "=== Data Processing Tool Demo ===" 
    putStrLn ""
    
    -- Process sample data
    let (employees, errors) = processEmployeeData sampleData
    
    putStrLn $ "Processed " ++ show (length sampleData) ++ " records"
    putStrLn $ "Valid: " ++ show (length employees) ++ ", Errors: " ++ show (length errors)
    putStrLn ""
    
    -- Show errors if any
    unless (null errors) $ do
        putStrLn "ERRORS:"
        mapM_ putStrLn errors
        putStrLn ""
    
    -- Generate and display report
    putStrLn $ processAndAnalyze sampleData
    
    -- Interactive analysis examples
    putStrLn "=== Department Analysis ==="
    putStrLn $ analyzeByDepartment "Engineering" employees
    
    putStrLn "=== High Earners (>$70k) ==="
    mapM_ (putStrLn . formatEmployee) (highEarners 70000 employees)
    
    putStrLn "=== Senior Employees (5+ years) ==="
    mapM_ (putStrLn . formatEmployee) (seniorEmployees 5 employees)
  where
    formatEmployee emp = empName emp ++ " - " ++ empDepartment emp ++ " - $" ++ 
                        printf "%.2f" (empSalary emp)
    unless condition action = if condition then return () else action

-- Utility function to demonstrate currying and partial application
createFilters :: Double -> Int -> String -> ([Employee] -> [Employee], [Employee] -> [Employee], [Employee] -> [Employee])
createFilters salaryThreshold yearThreshold dept = 
    (highEarners salaryThreshold, seniorEmployees yearThreshold, departmentFilter dept)

-- Example of using the filters
applyFilters :: [Employee] -> [Employee]
applyFilters employees = 
    let (salFilter, yearFilter, deptFilter) = createFilters 70000 5 "Engineering"
    in salFilter . yearFilter . deptFilter $ employees

-- This demonstrates the power of higher-order functions in creating
-- flexible, composable data processing pipelines