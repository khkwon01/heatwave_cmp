SELECT "-- Installation: STARTED ---" AS "";

SELECT "-- Dropping database if exists ---" AS "";

DROP DATABASE IF EXISTS ecommerce;

SELECT "-- Creating tables ---" AS "";

source create_tables.sql

SELECT "-- Populating data ---" AS "";

source create_data.sql

SELECT "-- Creating stored procedures ---" AS "";

source procedures.sql

SELECT "-- Installation: DONE ---" AS "";
