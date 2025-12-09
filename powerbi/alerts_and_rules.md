# Power BI Alerts and Rules

## Setting Up Alerts in Power BI Service

### Alert 1: High Fraud Rate

**Condition:** Fraud Rate > 5%
**Frequency:** Once per day
**Notification:** Email + Mobile

**DAX Measure:**
```
High Fraud Alert = IF([Fraud Rate] > 0.05, 1, 0)
```

### Alert 2: Critical Risk Transactions

**Condition:** Critical Risk Count > 10 in last hour
**Frequency:** Every hour
**Notification:** Email

**DAX Measure:**
```
Critical Risk Alert = 
IF(
    CALCULATE(
        [Critical Risk Count],
        FILTER(
            Transactions,
            Transactions[TransactionDate] >= NOW() - TIME(1, 0, 0)
        )
    ) > 10,
    1,
    0
)
```

### Alert 3: High Value Fraud Detected

**Condition:** Amount at Risk > $10,000 in last 24 hours
**Frequency:** Once per day
**Notification:** Email

**DAX Measure:**
```
High Value Alert = 
IF(
    CALCULATE(
        [Total Amount at Risk],
        FILTER(
            Transactions,
            Transactions[TransactionDate] >= TODAY() - 1
        )
    ) > 10000,
    1,
    0
)
```

## Business Rules

### Rule 1: Auto-Block Threshold

Transactions with fraud probability > 0.9 should be automatically blocked.

**Implementation:**
- Create calculated column: `AutoBlock = IF([fraud_probability] > 0.9, "BLOCK", "REVIEW")`

### Rule 2: Review Queue Priority

Prioritize review queue by:
1. Critical risk transactions
2. High value transactions (>$500)
3. Recent transactions (last 24 hours)

**DAX Measure:**
```
Review Priority = 
SWITCH(
    TRUE(),
    [RiskLevel] = "Critical", 1,
    [RiskLevel] = "High" && [amt] > 500, 2,
    [RiskLevel] = "High", 3,
    [RiskLevel] = "Medium" && [amt] > 500, 4,
    5
)
```

### Rule 3: Daily Fraud Limit

If fraud rate exceeds 15% in a single day, trigger escalation.

**DAX Measure:**
```
Daily Fraud Limit Exceeded = 
IF(
    CALCULATE(
        [Fraud Rate],
        FILTER(
            Transactions,
            Transactions[date] = TODAY()
        )
    ) > 0.15,
    1,
    0
)
```

## KPI Thresholds

- **Green (Good):** Fraud Rate < 2%
- **Yellow (Warning):** Fraud Rate 2-5%
- **Red (Critical):** Fraud Rate > 5%

## Conditional Formatting Rules

### Card Background Colors

```
IF([Fraud Rate] < 0.02, "#00B050",
IF([Fraud Rate] < 0.05, "#FFC000", "#C00000"))
```

### Table Row Colors

- Low Risk: White background
- Medium Risk: Light yellow (#FFF9E6)
- High Risk: Light orange (#FFE6CC)
- Critical Risk: Light red (#FFCCCC)
