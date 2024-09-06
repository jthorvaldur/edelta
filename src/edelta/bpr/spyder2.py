from basefunc import *


if __name__ == "__main__":
    delta, df = get_test_data_df()
    mult = 1

    rolling_vol = delta.rolling(60).std()
    rolling_vol[rolling_vol == 0] = np.nan
    rolling_vol = rolling_vol.ffill().bfill()
    rolling_vol_l1 = rolling_vol.shift(1).ffill().bfill()
    delta /= rolling_vol_l1
    tgt = delta.shift(-1).fillna(0) / rolling_vol
    # convert delta index to int64 increasing from 0 to len(delta)
    dt_idx = delta.index
    delta.index = np.arange(len(delta))

    n_b = 3
    count = n_b * 21 * mult
    nroll = count
    dt = count / n_b

    total_points_indep = int(len(delta) / count)
    total_points = int(len(delta) / nroll)
    print("total_points_indep", total_points_indep)
    print("total_points", total_points)
    print("count", count)
    # csp = index_csum(delta, count) # maybe ignore this, do at opt stage
    csp = delta.cumsum()
    basis = gencbasis_df(dt, count, n_b)
    print(basis.cov())

    if False:
        basis.plot(figsize=(11, 9))
        plt.show()
        sys.exit()

    # lr = LinearRegression(fit_intercept=False)
    lr = BayesianRidge(fit_intercept=False)
    # lr = LassoLars(fit_intercept=False)

    # csp = cumsum_by_day(delta.copy())

    # delta.cumsum().plot(figsize=(11, 9))
    # csp.plot(secondary_y=True, alpha=0.6, color="red")
    # plt.show()
    # sys.exit()

    # plt.show()

    weights = np.zeros((n_b + 0, total_points))
    volweights = np.zeros((1, total_points))
    dt_arr = []
    id_arr = []
    for kroll in list(range(total_points))[::]:
        istart = kroll * nroll
        iend = istart + count
        optx = csp.loc[istart : iend - 1] * 1
        xdelta = delta.iloc[istart : iend - 0]
        if len(optx) < count:
            break
        optx.index = basis.index
        optx -= optx.values[0]
        # optx -= optx.mean()
        lr.fit(basis, optx)
        weight = lr.coef_
        # weight = np.around(weight, 4)
        weights[:, kroll] = weight  # /(xdelta.std()*count**0.5)
        # weights[:-2,kroll] = weight
        # volweights["std",kroll] = xdelta.std()
        volweights[0, kroll] = xdelta.sum()  # /xdelta.std()
        # print(weight)
        base_delta = delta.iloc[istart : iend - 0].reset_index().delta.cumsum()
        base_delta -= base_delta[0]
        dt_arr.append(dt_idx[iend])
        id_arr.append(iend)
        # print(dt_idx)
        # print(len(base_delta),len(optx))

        if not False:
            cts = (weight * basis).sum(1)
            cts.plot(figsize=(11, 9), color="blue", lw=2, ls="-.")
            optx.plot(color="red", alpha=0.6)
            base_delta.plot(color="black", alpha=0.6)
            plt.show()
            sys.exit()

    dfw = pd.DataFrame(weights[:, :]).T

    # apply kmeans to dfw to find sample clusters
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=32).fit(dfw)
    labels = pd.Series(kmeans.labels_)

    dfw["label"] = labels
    print(dfw.groupby("label").mean())
    volweights = pd.DataFrame(volweights.T)
    fvolweights = volweights.shift(-1).fillna(0)
    volweights["label"] = labels
    fvolweights["label"] = labels

    vt = fvolweights.pivot(columns="label", values=0).fillna(0)
    vt_sharpe = vt.mean() / vt.std()
    # vt *= vt_sharpe
    vt.cumsum().plot(figsize=(11, 9))
    vt.mean(1).cumsum().plot(secondary_y=True, color="black", lw=2)
    plt.show()

    # print(volweights.groupby("label"))
    #
    # print(volweights.groupby("label").mean())
    # print(fvolweights.groupby("label").mean())

    # print(labels.value_counts())
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)
    sys.exit()
    print(dfw.shape)
    j = 0
    val = 0.0

    cdf = pd.DataFrame(csp)
    cdf["gen"] = np.nan
    # print(cdf)
    # sys.exit()
    for i in id_arr[0:]:
        # iend = i
        # istart = iend - count
        # base_delta = delta.iloc[istart:iend - 0].reset_index().delta.cumsum()
        # base_delta -= base_delta[0]
        # optx = csp.loc[istart:iend - 1] * 1
        # xdelta = delta.iloc[istart:iend - 0]
        # if len(optx) < count:
        #     break
        # optx.index = basis.index
        # optx -= optx.values[0]
        # cst = (dfw.iloc[j,:] * basis).sum(1)
        # cst.plot(figsize=(11, 9), color="blue", lw=2, ls="-.")
        # optx.plot(color="red", alpha=0.6)
        # base_delta.plot(color="black", alpha=0.6)
        # plt.show()
        # sys.exit()

        val += (dfw.iloc[j, :] * basis).sum(1).values[-1]
        # print(f"{dt_arr[j]} {i} {csp[i]:.2f} {val:.2f}")
        cdf.loc[i, "gen"] = val
        j += 1
    # cdf = cdf.ffill().bfill()
    cdf = cdf.dropna()
    cdf -= cdf.iloc[0]
    # cdf.plot(figsize=(11, 9))
    # plt.show()
    # sys.exit()

    # dfw = dfw.T
    dfw.index = dt_arr

    dfw.reset_index(drop=True).plot(figsize=(11, 9))
    plt.show()
    sys.exit()

    tgt_sub = tgt[dfw.index]
    tgt_sub = tgt_sub.reset_index(inplace=False, drop=False).delta
    dft = pd.DataFrame(dfw.values, columns=[f"w_{i}" for i in range(n_b)])
    idx = dft["w_0"].sort_values().index
    # print(tgt_sub)
    # print(tgt_sub.loc[idx].reset_index().delta.cumsum())
    # sys.exit()
    # tgt_sub.loc[idx].reset_index().delta.cumsum().plot(figsize=(11, 9))
    # tgt_sub.cumsum().plot(secondary_y=False, alpha=0.6, color="red")
    # plt.show()
    # sys.exit()
    # # print(tgt_sub)
    # tgt_sub.cumsum().plot(figsize=(11, 9))
    # tgt.reset_index().delta.cumsum().plot(secondary_y=True, alpha=0.6, color="red")
    # plt.show()
    # sys.exit()

    U, S, V = np.linalg.svd(dfw.cov())

    dfo = dfw @ U
    U[:, -1] *= 0
    print(dfw.corr())
    dfw = dfo @ U.T
    print(dfo.corr())
    print(dfw.corr())
    # print(dfw.tail())
    sys.exit()

    print(dfo.mean() / dfo.std())
    print(dfw.T.mean() / dfw.T.std())
    # sys.exit()
    # calc partial autocorrelation for each column
    df_pacf = pd.DataFrame(index=range(1, 6))
    df_pacf_o = pd.DataFrame(index=range(1, 6))
    for i in range(n_b):
        df_pacf[i] = pd.Series(sm.tsa.stattools.pacf(dfw.T[i], nlags=5))
        df_pacf_o[i] = pd.Series(sm.tsa.stattools.pacf(dfo[i], nlags=5))
        # print(f"pacf {i}: {pd.Series(sm.tsa.stattools.pacf(dfo[i],nlags=5))}")

    print(df_pacf)
    print(df_pacf_o)
    # dfo.plot.scatter(x=0,y=1)
    # plt.show()

    # dfo.plot(figsize=(11, 9),alpha=0.7,legend=False,ls="-",marker="*")
    # plt.show()
    sys.exit()

    S_ = pd.Series(S)
    S_ /= S_.sum()
    print(S_.head(21))
    # S_.plot(figsize=(11, 9))
    # S_.cumsum().plot()
    # plt.show()

    # print(dfw.corr())
    sys.exit()

    dvw = pd.DataFrame(volweights.T)

    dvw.columns = ["std", "ave"]
    dvw["std"].plot(figsize=(11, 9))
    plt.show()
    sys.exit()

    print(dfw.head())
    # dfw /= dfw.abs().sum(1)
    dfw = dfw.div(dvw[0], axis=0)
    print(dfw.head())
    dfw = (1 * dfw).astype(np.int32)
    print(dfw.nunique())
    # sys.exit()
    print(dfw.head(), dfw.shape)
    print(dvw.head(), dvw.shape)
    sys.exit()
    dfw = dfw  # + n_b//2 -1
    string_row = dfw.apply(lambda x: "".join(x.astype(str)).replace("-", "n"), axis=1)
    ust = string_row.unique()
    print(ust)
    print(f"{len(ust)} {len(string_row)} {len(ust)/len(string_row)}")

    sys.exit()
    # print(string_row)

    # print(dfw.drop_duplicates().sort_values(by=list(range(n_b))))
    # # print(dfw.nunique(axis=1))
    # l1 = list(dfw.itertuples(index=False))
    # s1 = set(l1)
    # # print(s1)
    # print(len(s1))
    # print(len(l1))

    dfw.to_csv("weights.csv")
    dfw.to_parquet("weights.parquet", compression="snappy")

    # dfw.columns = [f"w_{i}" for i in range(n_b)] + ["mean","std"]
    # dfw["w_0"].plot(figsize=(11, 9))
    # plt.show()
    # sys.exit()
    # dfw.columns = [f"w_{i}" for i in range(n_b)] + ["mean","std"]
    cov = dfw.cov()
    U, S, V = np.linalg.svd(cov)
    print(S)
    print(S / S.sum())

    # linear model for previous weights to predict next weights
    lr = LinearRegression(fit_intercept=False)
    lr.fit(dfw.iloc[:-1, :], dfw.iloc[1:, :])
    weights = lr.coef_
    print(weights)
    # print(weights/weights.sum())
    # make weighted df of previous weights
    # dfw_model = dfw *0
    # for i in range(n_b):
    #     dfw_model += dfw.iloc[:-1,i] * weights[i]
    # make raw next weights
    dfw_f = dfw.shift(1).fillna(0)
    pnl = dfw * dfw_f
    pnl.cumsum().plot(figsize=(11, 9))
    pnl.mean(1).cumsum().plot()
    plt.show()

    sys.exit()
    print(dfw)
    # dfw = dfw.sub(dfw.mean(axis=1), axis=0)
    print(dfw.corr())
    print(dfw.abs().mean())
    dfw[dfw.columns.values[-1]].plot(figsize=(11, 9))
    dfw[dfw.columns.values[-2]].plot(secondary_y=True, alpha=0.6, color="red")
    plt.show()
    sys.exit()

    optx = csp.loc[N_obs * count : (N_obs + 1) * count - 1]
    optx.index = basis.index
    print(optx)
    print(csp)
    # sys.exit()
    lr.fit(basis, optx)
    weights = lr.coef_
    weights = np.around(weights, 4)
    print(weights)
    cts = (weights * basis).sum(1)

    cts.plot(figsize=(11, 9))
    optx.plot()
    delta.iloc[
        N_obs * count : (N_obs + 1) * count - 1
    ].reset_index().delta.cumsum().plot()
    plt.show()
    # print(optx)
    # print(optx.shape)

    # print(csp.tail())
    # print(csp.shape)
    # csp[N_obs*count:(N_obs+1)*count].plot(figsize=(11, 9))
    # idxs[N_obs*count:(N_obs+1)*count].plot(alpha=0.6,color="red",secondary_y=True)
    # plt.show()

    sys.exit()

    print(df.iloc[100:1000:50])
    # print (df["sec"].value_counts())
    daycs = df.groupby(str(df.index.strftime("%D"))).cumsum()
    print(daycs.head())
    print(df.index.day)
    sys.exit()

    print(df.resample("1D").mean())
    print(df.resample("1D").mean().isnull().sum())
    print(df.isnull().sum())
    print(df.shape)
    sys.exit()

    delta.plot(figsize=(11, 9))
    plt.show()

    print(df.head())
    print(df.dtypes)
    print(df.describe())
