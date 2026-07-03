-- Monthly churn rate = customers who churned that month / active customers at start of month
WITH monthly_active AS (
    SELECT billing_month, COUNT(DISTINCT customer_id) AS active_customers
    FROM billing_events
    GROUP BY billing_month
),
monthly_churn AS (
    SELECT churn_date AS billing_month, COUNT(*) AS churned_customers
    FROM customers
    WHERE churn_date IS NOT NULL
    GROUP BY churn_date
)
SELECT
    a.billing_month,
    a.active_customers,
    COALESCE(c.churned_customers, 0)                                    AS churned_customers,
    ROUND(100.0 * COALESCE(c.churned_customers, 0) / a.active_customers, 3) AS churn_rate_pct
FROM monthly_active a
LEFT JOIN monthly_churn c ON a.billing_month = c.billing_month
ORDER BY a.billing_month;
