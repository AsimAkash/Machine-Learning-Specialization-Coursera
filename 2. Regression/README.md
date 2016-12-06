# Machine Learning Specialization - University of Washington

## Course 2 - Regression

### Case Study - Predicting Housing Prices

Regularized linear regression models were used for the task of prediction and feature selection. The model predicts a continuous value (price) from input features (square footage, number of bedrooms and bathrooms,...). Using large feature sets, models of varing complexity were built. The impact of outliers on models and predictions were examined. Optimization algorithms were implemented to scale to large datasets.

### Work done

* Compare and contrast bias and variance when modeling data.
* Estimate model parameters using optimization algorithms.
* Tune parameters with cross validation.
* Analyze the performance of the model.
* Describe the notion of sparsity and how LASSO leads to sparse solutions.
* Deploy methods to select between models.
* Exploit the model to form predictions.
* Build a regression model to predict prices using a housing dataset.
* Implement these techniques in Python.

### Datasets

* [kc_house_data.gl.zip](https://d3c33hcgiwev3.cloudfront.net/_026a0fd773a5fdd104e1a6ca3cfb2622_kc_house_data.gl.zip?Expires=1481155200&Signature=R1fyyPzw1bbjhCLYfhEKt-PYvYonbXxB~kDa7YLvXnud0TF850skaHXpQbz1UmRfSzRGWprxVY0WvYZ2i0QWAx81NElQIofxWulbw1EFqKBFkd8R7xBmsIIk4nbmejUlB6pgfLvJroQMnpUSDP3srpw1Z50ClfmI~i9Ef0Hgasc_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [kc_house_data_small.gl.zip](https://d3c33hcgiwev3.cloudfront.net/_657b21c41f5e3679cc495391bee3baef_kc_house_data_small.gl.zip?Expires=1481155200&Signature=On9Fx8eq8j37lBdGj9eNMCrGl6DshsYSosSMRe~Vvr-Ko-bPX95U7cYChNIRwr-ZVe4pLJnVXpm0dl5HeoA9ZtDKP44XyGFR1DSSHLHXj9aBdXH7-U~es9wKirLU2vsuYVc1kvBDLPXEVfad2UMPOsY3BrmFO9y43Afb3Gvajbw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [kc_house_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_46994807796a1213d2699c6d9a09667c_kc_house_data.csv.zip?Expires=1481155200&Signature=DKJoL~HdrnZi-OR31ylZ4ZpSHSCa4-aKDnA4E6pIU74bMV~MxV3gVeGaEbZK1J81hIt4XB9qTHWTpqQUkp5ohlPsMQYaqzVsvVElzpdjEKVFcUB8U4pYl4zudZVEzLkLehGLxPofr4zy7NnGlQWENzLp~1DUkSmhN8G7tl8-DU8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [kc_house_train_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_46994807796a1213d2699c6d9a09667c_kc_house_train_data.csv.zip?Expires=1481155200&Signature=NXcmsTgoER7c-~xC6eA-4frswwwCK0mTpwxzAOHAT4iq-i1WgcEIgvHxLJlbmkjPyM-WT0JolzcecnKtpBUksq7pRV7MhFiAXowDCk1FkpCyx4UnKWEZsL7eWosHedLehpSik4Z34t2e1mhgDyZydtN2T951R0HL4cFtlpNsuSI_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [kc_house_test_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_46994807796a1213d2699c6d9a09667c_kc_house_test_data.csv.zip?Expires=1481155200&Signature=Kcj~h~vL1OhvsCmanmdyFwlSJuHcizffzDY-7VwhNcznzv5c8bJ7F-~HUP-W5Hf-Lf3py7OwOCRIHxaA9AUv8Z6C82CUuf8JJwpvC1efIs-dWgIg6jnfhEAjuEmnNSMGFJhZep8SZPgfMtK-yF5tKokpKqCQReD34s~H~KGHrcM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [wk3_kc_house_train_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_f626f6faf3c1039d014563b39ede3037_wk3_kc_house_train_data.csv.zip?Expires=1481155200&Signature=VDld6T0xFUgzoT2lOAk5XAZ6w5MOLv1lsHbXSr7wtAy4nwe~W9ZXuJm8WSjQfRAyP6U6ZRImCQpcC~FOWL9mqAz9HEeYq93EWKCT4Ky3jHYCXsYoQi0kN5lHU1A8FL-PPyGYGsgWpj-yQ5XmS2b1qme5mo1T7zpML8GkhISmuHI_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [wk3_kc_house_valid_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_f626f6faf3c1039d014563b39ede3037_wk3_kc_house_valid_data.csv.zip?Expires=1481155200&Signature=bSOoJAj14eXTCC6I1Kavb1zO686NWRcmqyU-p0ZwPeXoRtJSzHOQ3lnLAD8SUmikj8v--vAiLsn~2qM5ghLcqYV6jQ6qlQ02F3paPnwz3R0HSryjMH44s-h6z2igDNn3vzMFzTiGY7Sfx8LvtiWNwyBVgX4eqhmcDABkMldQwKA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [wk3_kc_house_test_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_f626f6faf3c1039d014563b39ede3037_wk3_kc_house_test_data.csv.zip?Expires=1481155200&Signature=YHIHxUG8XEyT4yAyeCwgf6kfKdN7XXatVeVUT6P8XpF~oIBcimKcGVpElviGc~ejrQPfTmnMTwiHMFBw1N1XXbiGy5F6gc8Gi81rrQM47RQaA-sGRFX-9DJdgDwR9REst7xluVKr8rTB2TwG04xKdiTYjBzTiyDtd~6meXoupGg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [wk3_kc_house_train_valid_shuffled.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_a6784a0fcfb928b2f3fa1d0ccbf8ce53_wk3_kc_house_train_valid_shuffled.csv.zip?Expires=1481155200&Signature=LcBtQtctjwsH4K1Qg8EVtr4LfNjJict19MgEZCkBXzHSPEfiuIvAi2qg8H1gXtakLlf0o8q6cQJ9YtDY69hbx4Mzi-G2AAzD1TYXgTR8v9dpYtp14GGxwGnVoLumUXGndjNNJtethy6HTR9bong3YH-Q0yugm7gI2a46FvxXY5I_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [wk3_kc_house_set_1_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_f626f6faf3c1039d014563b39ede3037_wk3_kc_house_set_1_data.csv.zip?Expires=1481155200&Signature=j0-00mM9p9WW1kpLRPzzJPy~x7LnOb0ZVL3DNIiOQrG5CWFrS641lSTnwUGrlfFTLmYrllyl8pIMA1CZp091rj0MWDhjGJ9gSK~j~88zN-K4~uWB6cAG3aSsflVmBKYINNO2DLhIJar2Hs6b6b~s9ART4Tw6QBoBWYLSWbuXHOA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [wk3_kc_house_set_2_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_f626f6faf3c1039d014563b39ede3037_wk3_kc_house_set_2_data.csv.zip?Expires=1481155200&Signature=hprTidlB~wL47fP7hVMrAHjpIBhgKHQxkClvRifRwkaZP9uKo9EkEjnNlIcEvSZ7e97lSx~VnYIqSC-McE7Ue-JWzfhRst~WRLilgY6am2DCSbPf0oaE8wXyv3lRE5-lUsNbW2H4B5LnHzrrU5ZS1nCH7zaBubuuol7krAt~fSE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [wk3_kc_house_set_3_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_f626f6faf3c1039d014563b39ede3037_wk3_kc_house_set_3_data.csv.zip?Expires=1481155200&Signature=Z~ibAxCIye3oPLpdke56Iv83CzY81WxfmpkzlIVMQ4u1hot32AMRbJLxcg97RGpjGOPU4OP6Jwei2Fm-HWYm~C2Obo7XNFC3keyCrb0IDJ4lMS-pf1DEso17ik~lNpIQMh~QrtvxB0mQQvZUF1A0kL6hvMgnZi34DRkfuAU~D~c_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [wk3_kc_house_set_4_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_f626f6faf3c1039d014563b39ede3037_wk3_kc_house_set_4_data.csv.zip?Expires=1481155200&Signature=US3guKval3bTHuQxldkoDqwcUFHq8-JWD2c~w~w-hEeMPGOgE-YxPwSoyLjHdu1niKul0Ttfy-kOV-jpQjy8Io8WUn1nJpEaHe6yH5H9ehO85Ikv5vajcmIMg8844DBGmeUoJd2bsQ7Hp4AxW1r9O6GZJRpLITqDm-p3JB98-P0_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [kc_house_data_small.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_b8bd4f01fc6e1df2579d87edb630d0ea_kc_house_data_small.csv.zip?Expires=1481155200&Signature=TrUXKzpa9lfJvJ362y6BW7l3eUBAqxZvdEZS7zm0gLpdB5GAhcEO7y-ej3fGDmBNEGaHTwImPAqkbIwMbRlQjYakQrPntJqnusPyZLOxdbu28D9iL3BbEDvtFuO1aAB6sf0--2-D2b~ko-RiKn6zzbo0WHXErMspgdOa-EQ76jQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [kc_house_data_small_train.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_b8bd4f01fc6e1df2579d87edb630d0ea_kc_house_data_small_train.csv.zip?Expires=1481155200&Signature=GRugvACdLWbyOXMwv-58k~K9fpNW06JEflb2zuvNel~5i5GCAx4gbJ7r9jsiTUesgWvzLouqMy8tVKxkTK0NfQ7QkwyOV2avDGn2L0v025CvIyj~cmDixU6~waw0h1uRNnuGrtrz5pcylm~1Z2IuTZfTpv7RiD8VSAAF-bawa~I_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [kc_house_data_small_test.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_b8bd4f01fc6e1df2579d87edb630d0ea_kc_house_data_small_test.csv.zip?Expires=1481155200&Signature=A0h-P9VVP2ye-VBfXd~Prg3xkRsrAICBzd0Jx~tlN50tud8ewQ3zgP7CsKTqhcghNfqa07i-I9c6ts40htquT~dgdwqqCPI2UVYwcmR-9j6KCdVI8ejNRF8j3A6WPGKUgD22Gu2jCuTbnolEyfxh~Pqb5Pc2e-M9ww~Gb6MarvY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* [kc_house_data_small_validation.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_b8bd4f01fc6e1df2579d87edb630d0ea_kc_house_data_small_validation.csv.zip?Expires=1481155200&Signature=cYTcIfbhbAhFQYoDhviK8cqf7P23Qkvr7og7zEFllY9xWFgD6VbpkN7D4wTzum~u1ESkU~fMaSSw2K0-feZ~UxREfyXjtYREyVl-exlnPOolhvW2RlGVZwoJVx-ETgfzuQaGJx41Zfk1P62SsCbJLXcj7HZRPcL1ZtuWTBLODGA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
