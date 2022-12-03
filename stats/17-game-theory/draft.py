

#--------------------------------------------------------------------------
# TODO: Visualize when the licensees operated.
#--------------------------------------------------------------------------

# TODO: Plot entries and exits by month.
# grouper = pd.Grouper(key='exit_date', freq='M')
# cultivator_exits = cultivator_data.groupby(grouper)['exit'].count()


# TODO: Plot lifespans.
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hlines(
#     patients[df.event.values == 0], 0, df[df.event.values == 0].time, color="C3", label="Censored"
# )
# ax.hlines(
#     patients[df.event.values == 1], 0, df[df.event.values == 1].time, color="C7", label="Uncensored"
# )
# ax.scatter(
#     df[df.metastasized.values == 1].time,
#     patients[df.metastasized.values == 1],
#     color="k",
#     zorder=10,
#     label="Metastasized",
# )
# ax.set_xlim(left=0)
# ax.set_xlabel("Months since mastectomy")
# ax.set_yticks([])
# ax.set_ylabel("Subject")
# ax.set_ylim(-0.25, n_patients + 0.25)
# ax.legend(loc="center right")



#--------------------------------------------------------------------------
# TODO: Fit a Bayesian Cox's proportional hazard model.
# See:
#     - Bayesian Survival Analysis
#     https://docs.pymc.io/en/v3/pymc-examples/examples/survival_analysis/survival_analysis.html
#
#     - Censored Data Models
#     https://docs.pymc.io/en/v3/pymc-examples/examples/survival_analysis/censored_data.html
#--------------------------------------------------------------------------

import arviz as az
import pymc3 as pm

# Estimate a Bayesian Poisson model.

# coords = {"intervals": intervals}

# with pm.Model(coords=coords) as model:

#     lambda0 = pm.Gamma("lambda0", 0.01, 0.01, dims="intervals")
#     beta = pm.Normal("beta", 0, sigma=1000)
#     lambda_ = pm.Deterministic("lambda_", T.outer(T.exp(beta * df.metastasized), lambda0))
#     mu = pm.Deterministic("mu", exposure * lambda_)
#     obs = pm.Poisson("obs", mu, observed=death)


# Sample from the posterior distribution.
# n_samples = 1000
# n_tune = 1000
# with model:
#     idata = pm.sample(
#         n_samples,
#         tune=n_tune,
#         target_accept=0.99,
#         return_inferencedata=True,
#         random_seed=RANDOM_SEED,
#     )

# Identify the estimated effect on the hazard rate.
# np.exp(idata.posterior["beta"]).mean()

# Plot the posterior distribution.
# az.plot_posterior(idata, var_names=["beta"])

# base_hazard = idata.posterior["lambda0"]
# met_hazard = idata.posterior["lambda0"] * np.exp(idata.posterior["beta"])


# def cum_hazard(hazard):
#     return (interval_length * hazard).cumsum(axis=-1)


# def survival(hazard):
#     return np.exp(-cum_hazard(hazard))


# def get_mean(trace):
#     return trace.mean(("chain", "draw"))


# Visualize.
# fig, (hazard_ax, surv_ax) = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(16, 6))

# az.plot_hdi(
#     interval_bounds[:-1],
#     cum_hazard(base_hazard),
#     ax=hazard_ax,
#     smooth=False,
#     color="C0",
#     fill_kwargs={"label": "Had not metastasized"},
# )
# az.plot_hdi(
#     interval_bounds[:-1],
#     cum_hazard(met_hazard),
#     ax=hazard_ax,
#     smooth=False,
#     color="C1",
#     fill_kwargs={"label": "Metastasized"},
# )

# hazard_ax.plot(interval_bounds[:-1], get_mean(cum_hazard(base_hazard)), color="darkblue")
# hazard_ax.plot(interval_bounds[:-1], get_mean(cum_hazard(met_hazard)), color="maroon")

# hazard_ax.set_xlim(0, df.time.max())
# hazard_ax.set_xlabel("Months since mastectomy")
# hazard_ax.set_ylabel(r"Cumulative hazard $\Lambda(t)$")
# hazard_ax.legend(loc=2)

# az.plot_hdi(interval_bounds[:-1], survival(base_hazard), ax=surv_ax, smooth=False, color="C0")
# az.plot_hdi(interval_bounds[:-1], survival(met_hazard), ax=surv_ax, smooth=False, color="C1")

# surv_ax.plot(interval_bounds[:-1], get_mean(survival(base_hazard)), color="darkblue")
# surv_ax.plot(interval_bounds[:-1], get_mean(survival(met_hazard)), color="maroon")

# surv_ax.set_xlim(0, df.time.max())
# surv_ax.set_xlabel("Months since mastectomy")
# surv_ax.set_ylabel("Survival function $S(t)$")

# fig.suptitle("Bayesian survival model")
