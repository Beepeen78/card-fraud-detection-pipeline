// Power Query (M) Script for Fraud Detection Dashboard
// Copy this into Power BI Desktop → Get Data → Blank Query → Advanced Editor

let
    // ============================================
    // TRANSACTIONS TABLE
    // ============================================
    TransactionsSource = Csv.Document(
        File.Contents("powerbi/out/transactions_scored.csv"),
        [Delimiter=",", Columns=10, Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),
    TransactionsPromoted = Table.PromoteHeaders(TransactionsSource, [PromoteAllScalars=true]),
    TransactionsChangedType = Table.TransformColumnTypes(
        TransactionsPromoted,
        {
            {"fraud_probability", type number},
            {"fraud_prediction", Int64.Type},
            {"amt", type number},
            {"unix_time", Int64.Type}
        }
    ),
    
    // Add calculated columns
    TransactionsWithDate = Table.AddColumn(
        TransactionsChangedType,
        "TransactionDate",
        each DateTime.From(#datetime(1970, 1, 1, 0, 0, 0) + #duration(0, 0, 0, [unix_time])),
        type datetime
    ),
    TransactionsWithRisk = Table.AddColumn(
        TransactionsWithDate,
        "RiskLevel",
        each if [fraud_probability] < 0.1 then "Low"
             else if [fraud_probability] < 0.5 then "Medium"
             else if [fraud_probability] < 0.9 then "High"
             else "Critical",
        type text
    ),
    TransactionsWithHour = Table.AddColumn(
        TransactionsWithRisk,
        "Hour",
        each Time.Hour([TransactionDate]),
        Int64.Type
    ),
    TransactionsWithDayOfWeek = Table.AddColumn(
        TransactionsWithHour,
        "DayOfWeek",
        each Date.DayOfWeek([TransactionDate], Day.Monday),
        Int64.Type
    ),
    TransactionsWithMonth = Table.AddColumn(
        TransactionsWithDayOfWeek,
        "Month",
        each Date.Month([TransactionDate]),
        Int64.Type
    ),
    
    // ============================================
    // DAILY METRICS TABLE
    // ============================================
    MetricsSource = Csv.Document(
        File.Contents("powerbi/out/metrics_daily.csv"),
        [Delimiter=",", Columns=5, Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),
    MetricsPromoted = Table.PromoteHeaders(MetricsSource, [PromoteAllScalars=true]),
    MetricsChangedType = Table.TransformColumnTypes(
        MetricsPromoted,
        {
            {"date", type date},
            {"mean_prob", type number},
            {"predicted_fraud", Int64.Type},
            {"rows", Int64.Type}
        }
    ),
    
    // ============================================
    // FINAL OUTPUT
    // ============================================
    TransactionsFinal = TransactionsWithMonth
in
    TransactionsFinal

// ============================================
// ALTERNATIVE: Load both tables separately
// ============================================
// Create two separate queries:
// 1. "Transactions" - Use the TransactionsFinal part above
// 2. "DailyMetrics" - Use MetricsChangedType part above
