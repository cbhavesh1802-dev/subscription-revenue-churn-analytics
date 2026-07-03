-- Monthly Recurring Revenue (MRR) trend
SELECT
    billing_month,
    COUNT(DISTINCT customer_id) AS active_customers,
    ROUND(SUM(mrr), 2)          AS total_mrr
FROM billing_events
GROUP BY billing_month
ORDER BY billing_month;
