### 1. Heatwave cluster load status

```
SELECT /*+set_var(use_secondary_engine=off)*/ NAME, FORMAT(NROWS,0) as 'ROWS',
       FORMAT(SIZE_BYTES/1024/1024, 0) as SIZE_MB, LOAD_STATUS, LOAD_PROGRESS
FROM performance_schema.rpd_tables, performance_schema.rpd_table_id
WHERE rpd_tables.ID = rpd_table_id.ID ORDER BY NAME;
```

### 2. Check memory status

```
SELECT /*+set_var(use_secondary_engine=off)*/ ID,
       MEMORY_USAGE/1024/1024/1024 as SIZE_GB,
       BASEREL_MEMORY_USAGE/1024/1024/1024 as BASEREL,
       MEMORY_TOTAL/1024/1024/1024 as TOTAL_GB
FROM performance_schema.rpd_nodes;
```

### 3. Autopilot 생성된 DDL 확인 (작업 동일 세션)

```
SELECT log->>"$.sql" AS "Load Script" FROM sys.heatwave_load_report WHERE type = "sql" ORDER BY id;
```

### 4. Autopilot 에러 확인
```
SELECT log FROM sys.heatwave_autopilot_report WHERE type = 'warn' or type = 'error';
```
