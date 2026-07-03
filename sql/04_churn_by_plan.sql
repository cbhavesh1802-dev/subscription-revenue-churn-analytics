-- Churn rate by plan tier -- which plan retains customers best?
SELECT
    plan,
    COUNT(*)                                             AS total_customers,
    SUM(CASE WHEN is_active = 0 THEN 1 ELSE 0 END)        AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN is_active = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(mrr), 2)                                    AS avg_mrr
FROM customers
GROUP BY plan
ORDER BY churn_rate_pct DESC;
