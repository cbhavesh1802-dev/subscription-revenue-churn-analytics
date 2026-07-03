-- Cohort retention: for each signup month cohort, what % are still active
-- N months later (classic SaaS cohort retention curve)
WITH cohorts AS (
    SELECT
        customer_id,
        strftime('%Y-%m-01', signup_date) AS cohort_month
    FROM customers
),
activity AS (
    SELECT
        b.customer_id,
        c.cohort_month,
        CAST(
            (strftime('%Y', b.billing_month) - strftime('%Y', c.cohort_month)) * 12 +
            (strftime('%m', b.billing_month) - strftime('%m', c.cohort_month))
        AS INTEGER) AS months_since_signup
    FROM billing_events b
    JOIN cohorts c ON b.customer_id = c.customer_id
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_size
    FROM cohorts
    GROUP BY cohort_month
)
SELECT
    a.cohort_month,
    a.months_since_signup,
    COUNT(DISTINCT a.customer_id)                                              AS retained_customers,
    cs.cohort_size,
    ROUND(100.0 * COUNT(DISTINCT a.customer_id) / cs.cohort_size, 2)           AS retention_pct
FROM activity a
JOIN cohort_sizes cs ON a.cohort_month = cs.cohort_month
GROUP BY a.cohort_month, a.months_since_signup
ORDER BY a.cohort_month, a.months_since_signup;
