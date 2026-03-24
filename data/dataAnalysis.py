import pandas as pd
df = pd.read_csv("E:\ProJect\Civic Complaint\data\BBMP_Grievances_2023.csv")   # change filename if needed
print("=== COLUMNS ===")
print(df.columns.tolist())
print("\n=== FIRST 5 ROWS ===")
print(df.head(5).to_string())
print("\n=== CATEGORY DISTRIBUTION ===")
category_col = None
for col in df.columns:
    if "category" in col.lower() or "grievance" in col.lower() or "type" in col.lower():
        category_col = col
        print(df[col].value_counts().head(15))
        break
print(f"\nMain text column is probably: 'description' or 'complaint_text'")
df.to_csv("E:\ProJect\Civic Complaint\data\BBMP_cleaned.csv", index=False)  # we'll use this later