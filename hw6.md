HW6_Okonkwo_co2554
================
Camille Okonkwo
2024-03-20

# Prepare the data

``` r
hwdata2 = read.csv("data/hwdata2.csv")

hwdata2.1 = uncount(hwdata2, weights = count)

# re-factor variables
hwdata2.1$pared2 = relevel(as.factor(hwdata2.1$pared),
                            ref="0")

hwdata2.1$public2 = relevel(as.factor(hwdata2.1$public),
                            ref="0")

hwdata2.1$apply = as.factor(hwdata2.1$apply)
                  
table(hwdata2.1$apply)
```

    ## 
    ##   1   2   3 
    ## 220 140  40

# Fit the ordinal logistic model

``` r
library(VGAM)
```

    ## Loading required package: stats4

    ## Loading required package: splines

``` r
# proportional odds model
fit = vglm(apply ~ pared2 + public2,
           data = hwdata2.1,
           family = cumulative("logitlink",
                               parallel = TRUE))
```

    ## Warning in eval(slot(family, "initialize")): response should be ordinal---see
    ## ordered()

``` r
summary(fit)
```

    ## 
    ## Call:
    ## vglm(formula = apply ~ pared2 + public2, family = cumulative("logitlink", 
    ##     parallel = TRUE), data = hwdata2.1)
    ## 
    ## Coefficients: 
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept):1  0.38899    0.11638   3.342 0.000831 ***
    ## (Intercept):2  2.46422    0.18663  13.204  < 2e-16 ***
    ## pared21       -1.12141    0.26469  -4.237 2.27e-05 ***
    ## public21      -0.09897    0.28002  -0.353 0.723772    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Names of linear predictors: logitlink(P[Y<=1]), logitlink(P[Y<=2])
    ## 
    ## Residual deviance: 722.6738 on 796 degrees of freedom
    ## 
    ## Log-likelihood: -361.3369 on 796 degrees of freedom
    ## 
    ## Number of Fisher scoring iterations: 4 
    ## 
    ## No Hauck-Donner effect found in any of the estimates
    ## 
    ## 
    ## Exponentiated coefficients:
    ##   pared21  public21 
    ## 0.3258200 0.9057735

# Test the proportional odds assumption

``` r
# non- proportional odds model
fit2 = vglm(apply ~  pared2 + public2,
            data = hwdata2.1,
           family = cumulative("logitlink",
                               parallel = FALSE))
```

    ## Warning in eval(slot(family, "initialize")): response should be ordinal---see
    ## ordered()

``` r
summary(fit2)
```

    ## 
    ## Call:
    ## vglm(formula = apply ~ pared2 + public2, family = cumulative("logitlink", 
    ##     parallel = FALSE), data = hwdata2.1)
    ## 
    ## Coefficients: 
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept):1  0.36893    0.11728   3.146  0.00166 ** 
    ## (Intercept):2  2.57696    0.21915  11.759  < 2e-16 ***
    ## pared21:1     -1.15337    0.29340  -3.931 8.46e-05 ***
    ## pared21:2     -1.08246    0.37002  -2.925  0.00344 ** 
    ## public21:1     0.07544    0.29506   0.256  0.79819    
    ## public21:2    -0.73805    0.40241  -1.834  0.06664 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Names of linear predictors: logitlink(P[Y<=1]), logitlink(P[Y<=2])
    ## 
    ## Residual deviance: 718.656 on 794 degrees of freedom
    ## 
    ## Log-likelihood: -359.328 on 794 degrees of freedom
    ## 
    ## Number of Fisher scoring iterations: 5 
    ## 
    ## Warning: Hauck-Donner effect detected in the following estimate(s):
    ## '(Intercept):2'
    ## 
    ## 
    ## Exponentiated coefficients:
    ##  pared21:1  pared21:2 public21:1 public21:2 
    ##  0.3155729  0.3387625  1.0783614  0.4780438

``` r
lrtest(fit2, fit)
```

    ## Likelihood ratio test
    ## 
    ## Model 1: apply ~ pared2 + public2
    ## Model 2: apply ~ pared2 + public2
    ##   #Df  LogLik Df  Chisq Pr(>Chisq)
    ## 1 794 -359.33                     
    ## 2 796 -361.34  2 4.0178     0.1341

# Calcualte odds ratio & 95% CI of a lower rating

``` r
exp(cbind(OR = coef(fit), confint(fit)))
```

    ##                       OR     2.5 %     97.5 %
    ## (Intercept):1  1.4754944 1.1745538  1.8535412
    ## (Intercept):2 11.7542539 8.1533435 16.9455005
    ## pared21        0.3258200 0.1939443  0.5473669
    ## public21       0.9057735 0.5231965  1.5681024

# Estimate the probability of rating “very likely”

``` r
# predicted probabilities
newdata = data.frame(pared2=c("0","1"),
                     public2=c("1", "0"))
pred = predict(fit, newdata=newdata, "response")
cbind(newdata,pred)
```

    ##   pared2 public2         1         2          3
    ## 1      0       1 0.5720028 0.3421359 0.08586129
    ## 2      1       0 0.3246646 0.4682863 0.20704913

# Fit a multinomial logistic regression model to the data using “unlikely” as the reference category.

``` r
fit3 = vglm(apply ~  pared2 + public2,
            data = hwdata2.1,
           family = multinomial(refLevel = "1"))

summary(fit3)
```

    ## 
    ## Call:
    ## vglm(formula = apply ~ pared2 + public2, family = multinomial(refLevel = "1"), 
    ##     data = hwdata2.1)
    ## 
    ## Coefficients: 
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept):1  -0.5611     0.1246  -4.502 6.73e-06 ***
    ## (Intercept):2  -2.1038     0.2229  -9.436  < 2e-16 ***
    ## pared21:1       1.0233     0.3134   3.265 0.001095 ** 
    ## pared21:2       1.5221     0.4140   3.677 0.000236 ***
    ## public21:1     -0.3124     0.3349  -0.933 0.350887    
    ## public21:2      0.5839     0.4259   1.371 0.170330    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Names of linear predictors: log(mu[,2]/mu[,1]), log(mu[,3]/mu[,1])
    ## 
    ## Residual deviance: 719.0465 on 794 degrees of freedom
    ## 
    ## Log-likelihood: -359.5232 on 794 degrees of freedom
    ## 
    ## Number of Fisher scoring iterations: 5 
    ## 
    ## No Hauck-Donner effect found in any of the estimates
    ## 
    ## 
    ## Reference group is level  1  of the response

``` r
# odds ratio and 95% CI
exp(cbind(OR=coef(fit3),confint(fit3)))
```

    ##                      OR      2.5 %     97.5 %
    ## (Intercept):1 0.5705826 0.44692543  0.7284538
    ## (Intercept):2 0.1219924 0.07880669  0.1888437
    ## pared21:1     2.7822510 1.50526080  5.1425776
    ## pared21:2     4.5817122 2.03538787 10.3135560
    ## public21:1    0.7317093 0.37958875  1.4104698
    ## public21:2    1.7930256 0.77821742  4.1311602

``` r
# predicted probabilities
d_hwdata2.1 = data.frame(pared2=rep(c("0","1"),2),
                         public2=rep(c("1","0"),2))
                     
pred2 = predict(fit3, newdata=d_hwdata2.1, "response")

cbind(d_hwdata2.1,pred2)
```

    ##   pared2 public2         1         2         3
    ## 1      0       1 0.6111587 0.2551591 0.1336821
    ## 2      1       0 0.3178197 0.5045400 0.1776403
    ## 3      0       1 0.6111587 0.2551591 0.1336821
    ## 4      1       0 0.3178197 0.5045400 0.1776403
