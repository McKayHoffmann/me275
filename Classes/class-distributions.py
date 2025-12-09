import scipy.stats as stats

mu = 50
std = 5
stats.norm.cdf(52, mu, std)
stats.norm.pdf(52, mu, std)

P2 = stats.norm.cdf(52, mu, std) - stats.norm.cdf(42, mu, std)

stats.norm.ppf(0.6554217416103242, mu, std)
stats.norm.pdf(0)
stats.norm.cdf(0)