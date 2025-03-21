### Collection of functions for synthetic data generation ###
While there are many datasets readily available, generating your own synthetic dataset allows benchmark model performance and test resilience against different patterns.
This can include:
- high dimensionality outlier: abnormal transactions that can be distinguished from other other similar transactions within its grouping
- behaviour: transactions may not stand out but the specific behaviour of the transaction relating to other transactions is abnormal
- collective pattern: an external event that triggers fraud-like transaction patterns but are not actually fraud (i.e.: A new concert tour of a famous star, a major gatcha game releasing hyped content, etc)
