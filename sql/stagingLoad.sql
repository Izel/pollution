LOAD DATA INFILE '/var/lib/mysql-files/airAnalysisData/LdnMetaStations.csv'
INTO TABLE ST_AURN_STATIONS_META
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/var/lib/mysql-files/airAnalysisData/2013.csv'
INTO TABLE ST_AURN_MEASURES_2013
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/var/lib/mysql-files/airAnalysisData/2014.csv'
INTO TABLE ST_AURN_MEASURES_2014
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/var/lib/mysql-files/airAnalysisData/2015.csv'
INTO TABLE ST_AURN_MEASURES_2015
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/var/lib/mysql-files/airAnalysisData/2016.csv'
INTO TABLE ST_AURN_MEASURES_2016
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/var/lib/mysql-files/airAnalysisData/2017.csv'
INTO TABLE ST_AURN_MEASURES_2017
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/var/lib/mysql-files/airAnalysisData/2018.csv'
INTO TABLE ST_AURN_MEASURES_2018
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
