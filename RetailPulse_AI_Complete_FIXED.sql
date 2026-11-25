CREATE DATABASE retailpulse_ai;
USE retailpulse_ai;
CREATE TABLE stores (
    store_id VARCHAR(10) PRIMARY KEY,
    store_name VARCHAR(100),
    location VARCHAR(100)
) ENGINE=InnoDB;

CREATE TABLE suppliers (
    supplier_id VARCHAR(10) PRIMARY KEY,
    supplier_name VARCHAR(100),
    region VARCHAR(50)
) ENGINE=InnoDB;

CREATE TABLE products (
    product_id VARCHAR(10) PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    unit_cost DECIMAL(12,2),
    unit_price DECIMAL(12,2),
    reorder_point INT,
    lead_time_days INT
) ENGINE=InnoDB;

CREATE TABLE sales_transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    transaction_date DATE,
    store_id VARCHAR(10),
    product_id VARCHAR(10),
    quantity INT,
    unit_price DECIMAL(12,2),
    discount_pct DECIMAL(8,2),
    revenue DECIMAL(12,2),
    profit DECIMAL(12,2),
    payment_method VARCHAR(30),
    CONSTRAINT fk_sales_store FOREIGN KEY (store_id)
        REFERENCES stores(store_id),
    CONSTRAINT fk_sales_product FOREIGN KEY (product_id)
        REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE purchase_orders (
    po_id VARCHAR(20) PRIMARY KEY,
    order_date DATE,
    store_id VARCHAR(10),
    product_id VARCHAR(10),
    supplier_id VARCHAR(10),
    ordered_quantity INT,
    received_quantity INT,
    delivery_delay_days INT,
    CONSTRAINT fk_po_store FOREIGN KEY (store_id)
        REFERENCES stores(store_id),
    CONSTRAINT fk_po_product FOREIGN KEY (product_id)
        REFERENCES products(product_id),
    CONSTRAINT fk_po_supplier FOREIGN KEY (supplier_id)
        REFERENCES suppliers(supplier_id)
) ENGINE=InnoDB;

CREATE TABLE inventory_snapshot (
    snapshot_date DATE,
    store_id VARCHAR(10),
    product_id VARCHAR(10),
    stock_on_hand INT,
    reorder_point INT,
    PRIMARY KEY (snapshot_date, store_id, product_id),
    CONSTRAINT fk_inventory_store FOREIGN KEY (store_id)
        REFERENCES stores(store_id),
    CONSTRAINT fk_inventory_product FOREIGN KEY (product_id)
        REFERENCES products(product_id)
) ENGINE=InnoDB;
SET @idx_exists = (
    SELECT COUNT(*) FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'sales_transactions'
      AND index_name = 'idx_sales_date'
);
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX idx_sales_date ON sales_transactions(transaction_date)',
    'SELECT ''idx_sales_date already exists - skipped'' AS index_status');
PREPARE retailpulse_stmt FROM @sql;
EXECUTE retailpulse_stmt;
DEALLOCATE PREPARE retailpulse_stmt;

SET @idx_exists = (
    SELECT COUNT(*) FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'sales_transactions'
      AND index_name = 'idx_sales_store'
);
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX idx_sales_store ON sales_transactions(store_id)',
    'SELECT ''idx_sales_store already exists - skipped'' AS index_status');
PREPARE retailpulse_stmt FROM @sql;
EXECUTE retailpulse_stmt;
DEALLOCATE PREPARE retailpulse_stmt;

SET @idx_exists = (
    SELECT COUNT(*) FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'sales_transactions'
      AND index_name = 'idx_sales_product'
);
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX idx_sales_product ON sales_transactions(product_id)',
    'SELECT ''idx_sales_product already exists - skipped'' AS index_status');
PREPARE retailpulse_stmt FROM @sql;
EXECUTE retailpulse_stmt;
DEALLOCATE PREPARE retailpulse_stmt;

SET @idx_exists = (
    SELECT COUNT(*) FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'purchase_orders'
      AND index_name = 'idx_po_supplier'
);
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX idx_po_supplier ON purchase_orders(supplier_id)',
    'SELECT ''idx_po_supplier already exists - skipped'' AS index_status');
PREPARE retailpulse_stmt FROM @sql;
EXECUTE retailpulse_stmt;
DEALLOCATE PREPARE retailpulse_stmt;

SET @idx_exists = (
    SELECT COUNT(*) FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'purchase_orders'
      AND index_name = 'idx_po_date'
);
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX idx_po_date ON purchase_orders(order_date)',
    'SELECT ''idx_po_date already exists - skipped'' AS index_status');
PREPARE retailpulse_stmt FROM @sql;
EXECUTE retailpulse_stmt;
DEALLOCATE PREPARE retailpulse_stmt;

SET @idx_exists = (
    SELECT COUNT(*) FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'inventory_snapshot'
      AND index_name = 'idx_inventory_product'
);
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX idx_inventory_product ON inventory_snapshot(product_id)',
    'SELECT ''idx_inventory_product already exists - skipped'' AS index_status');
PREPARE retailpulse_stmt FROM @sql;
EXECUTE retailpulse_stmt;
DEALLOCATE PREPARE retailpulse_stmt;

SHOW TABLES;

SELECT 'stores' AS table_name, COUNT(*) AS total_rows FROM stores
UNION ALL
SELECT 'suppliers', COUNT(*) FROM suppliers
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'sales_transactions', COUNT(*) FROM sales_transactions
UNION ALL
SELECT 'purchase_orders', COUNT(*) FROM purchase_orders
UNION ALL
SELECT 'inventory_snapshot', COUNT(*) FROM inventory_snapshot;

SELECT
    SUM(transaction_id IS NULL) AS missing_transaction_id,
    SUM(transaction_date IS NULL) AS missing_transaction_date,
    SUM(store_id IS NULL) AS missing_store_id,
    SUM(product_id IS NULL) AS missing_product_id,
    SUM(quantity IS NULL) AS missing_quantity,
    SUM(revenue IS NULL) AS missing_revenue
FROM sales_transactions;

SELECT *
FROM sales_transactions
WHERE quantity <= 0 OR revenue < 0;

SELECT *
FROM purchase_orders
WHERE received_quantity > ordered_quantity
   OR received_quantity < 0
   OR ordered_quantity < 0;

SELECT *
FROM inventory_snapshot
WHERE stock_on_hand < 0;

SELECT
    COUNT(*) AS total_transactions,
    SUM(quantity) AS total_units_sold,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND(AVG(revenue), 2) AS average_transaction_revenue
FROM sales_transactions;

SELECT
    DATE_FORMAT(transaction_date, '%Y-%m') AS sales_month,
    COUNT(*) AS total_transactions,
    SUM(quantity) AS units_sold,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(SUM(profit), 2) AS total_profit
FROM sales_transactions
GROUP BY DATE_FORMAT(transaction_date, '%Y-%m')
ORDER BY sales_month;

SELECT
    p.product_id,
    p.product_name,
    p.category,
    SUM(s.quantity) AS units_sold,
    ROUND(SUM(s.revenue), 2) AS total_revenue,
    ROUND(SUM(s.profit), 2) AS total_profit
FROM sales_transactions s
JOIN products p ON p.product_id = s.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_revenue DESC
LIMIT 10;

SELECT
    st.store_id,
    st.store_name,
    COUNT(s.transaction_id) AS total_transactions,
    COALESCE(SUM(s.quantity), 0) AS units_sold,
    ROUND(COALESCE(SUM(s.revenue), 0), 2) AS total_revenue,
    ROUND(COALESCE(SUM(s.profit), 0), 2) AS total_profit
FROM stores st
LEFT JOIN sales_transactions s ON s.store_id = st.store_id
GROUP BY st.store_id, st.store_name
ORDER BY total_revenue DESC;

SELECT
    payment_method,
    COUNT(*) AS transaction_count,
    ROUND(SUM(revenue), 2) AS total_revenue
FROM sales_transactions
GROUP BY payment_method
ORDER BY total_revenue DESC;

-- 9. INVENTORY INTELLIGENCE
SELECT
    i.snapshot_date,
    SUM(i.stock_on_hand) AS total_units_in_stock,
    ROUND(SUM(i.stock_on_hand * p.unit_cost), 2) AS inventory_cost_value,
    SUM(i.stock_on_hand <= i.reorder_point) AS low_stock_records
FROM inventory_snapshot i
JOIN products p ON p.product_id = i.product_id
GROUP BY i.snapshot_date
ORDER BY i.snapshot_date DESC;

SELECT
    i.snapshot_date,
    i.store_id,
    i.product_id,
    p.product_name,
    p.category,
    i.stock_on_hand,
    i.reorder_point,
    GREATEST(i.reorder_point - i.stock_on_hand, 0) AS suggested_minimum_reorder
FROM inventory_snapshot i
JOIN products p ON p.product_id = i.product_id
WHERE i.stock_on_hand <= i.reorder_point
ORDER BY suggested_minimum_reorder DESC;

SELECT
    p.category,
    SUM(i.stock_on_hand) AS total_stock_units,
    ROUND(SUM(i.stock_on_hand * p.unit_cost), 2) AS inventory_cost_value
FROM inventory_snapshot i
JOIN products p ON p.product_id = i.product_id
GROUP BY p.category
ORDER BY inventory_cost_value DESC;

SELECT
    store_id,
    COUNT(*) AS inventory_records,
    SUM(stock_on_hand <= reorder_point) AS low_stock_records
FROM inventory_snapshot
GROUP BY store_id
ORDER BY low_stock_records DESC;

-- 10. SUPPLIER INTELLIGENCE
SELECT
    sp.supplier_id,
    sp.supplier_name,
    COUNT(po.po_id) AS total_orders,
    COALESCE(SUM(po.ordered_quantity), 0) AS total_ordered_quantity,
    COALESCE(SUM(po.received_quantity), 0) AS total_received_quantity,
    COALESCE(SUM(po.ordered_quantity - po.received_quantity), 0) AS total_short_quantity,
    ROUND(AVG(po.delivery_delay_days), 2) AS average_delay_days,
    SUM(po.delivery_delay_days <= 0) AS on_time_orders,
    SUM(po.delivery_delay_days > 0) AS delayed_orders,
    ROUND(100 * SUM(po.delivery_delay_days <= 0) /
        NULLIF(COUNT(po.po_id), 0), 2) AS on_time_delivery_pct,
    ROUND(100 * SUM(po.received_quantity) /
        NULLIF(SUM(po.ordered_quantity), 0), 2) AS fulfilment_pct
FROM suppliers sp
LEFT JOIN purchase_orders po ON po.supplier_id = sp.supplier_id
GROUP BY sp.supplier_id, sp.supplier_name
ORDER BY on_time_delivery_pct DESC;

SELECT
    po.po_id,
    sp.supplier_name,
    po.product_id,
    po.ordered_quantity,
    po.received_quantity,
    (po.ordered_quantity - po.received_quantity) AS short_quantity,
    po.delivery_delay_days
FROM purchase_orders po
JOIN suppliers sp ON sp.supplier_id = po.supplier_id
WHERE po.received_quantity < po.ordered_quantity
ORDER BY short_quantity DESC;

-- 11. BUSINESS DECISION INTELLIGENCE
-- Aggregate sales and inventory separately to avoid multiplying rows
-- when one product has many sales transactions and several store stocks.
SELECT
    p.product_id,
    p.product_name,
    COALESCE(sales.total_units_sold, 0) AS total_units_sold,
    COALESCE(stock.current_stock, 0) AS current_stock,
    stock.reorder_point
FROM products p
JOIN (
    SELECT product_id, SUM(stock_on_hand) AS current_stock,
           MAX(reorder_point) AS reorder_point
    FROM inventory_snapshot
    GROUP BY product_id
) stock ON stock.product_id = p.product_id
LEFT JOIN (
    SELECT product_id, SUM(quantity) AS total_units_sold
    FROM sales_transactions
    GROUP BY product_id
) sales ON sales.product_id = p.product_id
WHERE stock.current_stock <= stock.reorder_point
ORDER BY total_units_sold DESC;

SELECT
    sp.supplier_name,
    COUNT(po.po_id) AS incomplete_orders,
    SUM(po.ordered_quantity - po.received_quantity) AS total_short_quantity,
    ROUND(AVG(po.delivery_delay_days), 2) AS average_delay_days
FROM suppliers sp
JOIN purchase_orders po ON po.supplier_id = sp.supplier_id
WHERE po.received_quantity < po.ordered_quantity
GROUP BY sp.supplier_id, sp.supplier_name
ORDER BY total_short_quantity DESC;

-- 12. REUSABLE VIEWS FOR POWER BI / TABLEAU
CREATE OR REPLACE VIEW vw_monthly_sales AS
SELECT
    DATE_FORMAT(transaction_date, '%Y-%m') AS sales_month,
    COUNT(*) AS total_transactions,
    SUM(quantity) AS units_sold,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(SUM(profit), 2) AS total_profit
FROM sales_transactions
GROUP BY DATE_FORMAT(transaction_date, '%Y-%m');

CREATE OR REPLACE VIEW vw_supplier_performance AS
SELECT
    sp.supplier_id,
    sp.supplier_name,
    COUNT(po.po_id) AS total_orders,
    SUM(po.ordered_quantity) AS total_ordered_quantity,
    SUM(po.received_quantity) AS total_received_quantity,
    SUM(po.ordered_quantity - po.received_quantity) AS total_short_quantity,
    ROUND(AVG(po.delivery_delay_days), 2) AS average_delay_days,
    ROUND(100 * SUM(po.delivery_delay_days <= 0) /
        NULLIF(COUNT(po.po_id), 0), 2) AS on_time_delivery_pct,
    ROUND(100 * SUM(po.received_quantity) /
        NULLIF(SUM(po.ordered_quantity), 0), 2) AS fulfilment_pct
FROM suppliers sp
LEFT JOIN purchase_orders po ON po.supplier_id = sp.supplier_id
GROUP BY sp.supplier_id, sp.supplier_name;

CREATE OR REPLACE VIEW vw_low_stock AS
SELECT
    i.snapshot_date,
    i.store_id,
    i.product_id,
    p.product_name,
    p.category,
    i.stock_on_hand,
    i.reorder_point,
    GREATEST(i.reorder_point - i.stock_on_hand, 0) AS suggested_minimum_reorder
FROM inventory_snapshot i
JOIN products p ON p.product_id = i.product_id
WHERE i.stock_on_hand <= i.reorder_point;

SELECT 'stores' AS table_name, COUNT(*) AS total_rows FROM stores
UNION ALL
SELECT 'suppliers', COUNT(*) FROM suppliers
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'sales_transactions', COUNT(*) FROM sales_transactions
UNION ALL
SELECT 'purchase_orders', COUNT(*) FROM purchase_orders
UNION ALL
SELECT 'inventory_snapshot', COUNT(*) FROM inventory_snapshot;

SELECT *
FROM sales_transactions
LIMIT 10;

DESCRIBE sales_transactions;
SHOW COLUMNS FROM sales_transactions;

SELECT
    COUNT(*) AS total_transactions,
    SUM(quantity) AS total_quantity_sold,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND(AVG(revenue), 2) AS average_transaction_value
FROM sales_transactions;
SELECT
    p.product_id,
    p.product_name,
    SUM(s.quantity) AS total_quantity_sold,
    ROUND(SUM(s.revenue), 2) AS total_revenue,
    ROUND(SUM(s.profit), 2) AS total_profit
FROM sales_transactions s
JOIN products p
    ON s.product_id = p.product_id
GROUP BY p.product_id, p.product_name
ORDER BY total_revenue DESC;
SELECT
    DATE_FORMAT(transaction_date, '%Y-%m') AS sales_month,
    COUNT(*) AS total_transactions,
    SUM(quantity) AS total_quantity,
    ROUND(SUM(revenue), 2) AS monthly_revenue,
    ROUND(SUM(profit), 2) AS monthly_profit
FROM sales_transactions
GROUP BY DATE_FORMAT(transaction_date, '%Y-%m')
ORDER BY sales_month;

SELECT
    store_id,
    COUNT(*) AS total_transactions,
    SUM(quantity) AS total_quantity_sold,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(SUM(profit), 2) AS total_profit
FROM sales_transactions
GROUP BY store_id
ORDER BY total_revenue DESC;

SELECT
    payment_method,
    COUNT(*) AS total_transactions,
    SUM(quantity) AS total_quantity,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(AVG(revenue), 2) AS average_transaction_value
FROM sales_transactions
GROUP BY payment_method
ORDER BY total_revenue DESC;

SELECT
    COUNT(*) AS total_inventory_records,
    SUM(stock_on_hand) AS total_stock_units,
    ROUND(SUM(stock_on_hand * p.unit_cost), 2) AS total_inventory_value,
    ROUND(AVG(stock_on_hand), 2) AS average_stock_per_record
FROM inventory_snapshot i
JOIN products p
    ON i.product_id = p.product_id;
    
SELECT
    i.store_id,
    i.product_id,
    p.product_name,
    p.category,
    i.stock_on_hand,
    i.reorder_point,
    (i.reorder_point - i.stock_on_hand) AS stock_shortage
FROM inventory_snapshot i
JOIN products p
    ON i.product_id = p.product_id
WHERE i.stock_on_hand <= i.reorder_point
ORDER BY stock_shortage DESC;

SELECT
    p.category,
    COUNT(*) AS inventory_records,
    SUM(i.stock_on_hand) AS total_stock,
    ROUND(SUM(i.stock_on_hand * p.unit_cost), 2) AS inventory_value,
    SUM(CASE
        WHEN i.stock_on_hand <= i.reorder_point THEN 1
        ELSE 0
    END) AS low_stock_records
FROM inventory_snapshot i
JOIN products p
    ON i.product_id = p.product_id
GROUP BY p.category
ORDER BY inventory_value DESC;

SELECT
    s.supplier_id,
    s.supplier_name,
    COUNT(po.po_id) AS total_orders,
    SUM(po.ordered_quantity) AS total_ordered,
    SUM(po.received_quantity) AS total_received,
    ROUND(AVG(po.delivery_delay_days), 2) AS avg_delivery_delay
FROM purchase_orders po
JOIN suppliers s
    ON po.supplier_id = s.supplier_id
GROUP BY s.supplier_id, s.supplier_name
ORDER BY avg_delivery_delay ASC;

SELECT
    CASE
        WHEN received_quantity = ordered_quantity THEN 'Completed'
        WHEN received_quantity = 0 THEN 'Not Received'
        ELSE 'Partially Received'
    END AS delivery_status,
    COUNT(*) AS total_orders,
    SUM(ordered_quantity) AS total_ordered,
    SUM(received_quantity) AS total_received
FROM purchase_orders
GROUP BY delivery_status
ORDER BY total_orders DESC;

SELECT
    s.supplier_id,
    s.supplier_name,
    SUM(po.ordered_quantity) AS total_ordered,
    SUM(po.received_quantity) AS total_received,
    SUM(po.ordered_quantity - po.received_quantity) AS short_quantity,
    ROUND(
        SUM(po.received_quantity) * 100.0 /
        NULLIF(SUM(po.ordered_quantity), 0),
        2
    ) AS fulfilment_percentage
FROM purchase_orders po
JOIN suppliers s
    ON po.supplier_id = s.supplier_id
GROUP BY s.supplier_id, s.supplier_name
ORDER BY fulfilment_percentage DESC;

DESCRIBE purchase_orders;

SELECT
    i.store_id,
    i.product_id,
    p.product_name,
    i.stock_on_hand,
    i.reorder_point,
    (i.reorder_point - i.stock_on_hand) AS recommended_reorder_quantity,
    p.lead_time_days
FROM inventory_snapshot i
JOIN products p
    ON i.product_id = p.product_id
WHERE i.stock_on_hand <= i.reorder_point
ORDER BY recommended_reorder_quantity DESC;

SHOW FULL TABLES
WHERE TABLE_TYPE = 'VIEW';

SELECT * FROM vw_monthly_sales LIMIT 10;

SELECT * FROM vw_supplier_performance;

SELECT * FROM vw_low_stock LIMIT 10;