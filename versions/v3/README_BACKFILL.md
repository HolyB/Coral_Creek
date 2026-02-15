# Backfill Acceleration Guide (Multi-Machine)

To speed up the backfill, you can run multiple instances on different machines (or processes) covering different date ranges.

## Machine Setup

1.  **Pull Latest Code**:
    ```bash
    git pull
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables**:
    Create a `.env` file or export variables:
    ```bash
    export POLYGON_API_KEY="YOUR_KEY"
    export SUPABASE_URL="..."
    export SUPABASE_KEY="..."
    # Optional: adjust workers if CPU allows (default is 12)
    # export MAX_WORKERS=20 
    ```

## Execution Strategy

**Machine A (Current)**:
Running range: `2022-01-01` to `2023-12-31`
(Keep this running)

**Machine B (New)**:
Run the following command to cover the recent years:
```bash
# Windows PowerShell
$env:PYTHONIOENCODING='utf-8'
python scripts/historical_backfill.py --phase 2 --market US --start 2024-01-01 --end 2026-02-15
```

## Important Notes

*   **Database**: Both machines will write to their local SQLite `db/coral_creek.db`.
*   **Merge**: After both finish, you will need to merge the SQLite databases or simply rely on the Supabase sync (Phase 3) to consolidate data in the cloud.
*   **Cache**: Each machine builds its own local Parquet cache. This is fine.
*   **ML Model**: ML generation is currently disabled to prevent crashes. Just focus on data backfill.

Good luck!
