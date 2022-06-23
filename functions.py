import pandas as pd
from os import environ
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from random import gauss
from random import seed
from pandas import Series
from pmdarima import auto_arima
import itertools
from time import time
sns.set_style('whitegrid')


def convert_to_weekly(signal):
    temp_signal = signal.copy()
    years = temp_signal.ds.dt.year.unique()
    start_year = years[0]
    end_year = years[-1]
    diff = end_year - start_year + 1
    res = []
    year = pd.DataFrame()
    for i in range(diff):
        year = temp_signal.loc[temp_signal.ds.dt.year == start_year + i]
        for j in range(365 // 7):
            k = (j + 1) * 7
            if k > 365-7:
                per = 365 - k + 7
            else:
                per = 7
            t = year.head(per)
            res.append(t.y.mean())
            year = year.drop(t.index, axis=0)
        # a.append(i)
    # a=[]
    # for i in range(365//7):
    #     k = (i+1)*7
    #
    #     if k > 365-7:
    #         per = 365 - k + 7
    #     else:
    #         per = 7
    #     a.append(per)
    # le
    # res.reset_index(inplace=True, drop=True)
    return pd.DataFrame(res).rename(columns={0: 'y'})


def filter_29_february(signal):
    filter = signal.loc[(signal.ds.dt.day == 29) & (signal.ds.dt.month == 2)]
    signal = signal.drop(filter.index, axis=0)
    signal.reset_index(inplace=True, drop=True)
    return signal


def define_trend(signal):
    years = signal.ds.dt.year.unique()
    start_year = years[0]
    end_year = years[-1]
    diff = end_year - start_year + 1
    trend = []
    mean = 0
    for i in range(365):
        for j in range(diff):
            mean = mean + signal.y[i + j*365]
        trend.append(mean/diff)
        mean = 0
    return pd.DataFrame(trend).rename(columns={0: 'y'})


def trend(signal, year_trend):
    temp_signal = signal.copy()
    years = temp_signal.ds.dt.year.unique()
    start_year = years[0]
    end_year = years[-1]
    diff = end_year - start_year + 1
    res = pd.DataFrame()
    for i in range(diff):
        res = pd.concat([res, year_trend])

    res.reset_index(inplace=True, drop=True)
    res['ds'] = temp_signal.ds
    return res


def remove_trend(signal, year_trend):
    temp_signal = signal.copy()
    years = temp_signal.ds.dt.year.unique()
    start_year = years[0]
    end_year = years[-1]
    diff = end_year - start_year + 1
    res = pd.DataFrame()
    for i in range(diff):
        res = pd.concat([res, year_trend])

    res.reset_index(inplace=True, drop=True)

    full_trend = trend(temp_signal, year_trend)
    res = pd.DataFrame(temp_signal.y-full_trend.y)
    temp_signal.y = res.y
    return temp_signal


def get_corelation_data(signal, L, h):
    m_dash = signal.y.mean()
    sum = 0
    a = []
    length = signal.y.shape[0] - L
    for j in range(L):
        for n in range(length-1):
            sum = sum + (signal.y[n*h] - m_dash)*(signal.y[(n+j)*h] - m_dash)
        a.append(sum/length)
        sum=0
#     return pd.DataFrame(a)
    return pd.DataFrame(a).rename(columns={0: 'y'})


def W(n, h, Um):
    return (0.53836 + 0.46164 * np.cos(np.pi * n * h / Um)) if n * h <= Um else 0


def spectral_density(R, L, wmax, wmin, omegaStepsAmount, h):
    add = 1 if omegaStepsAmount % 2 == 0 else 0
    res = []
    fres = []
    sum = 0
    omegaStep = (wmax - wmin) / (omegaStepsAmount + add - 1)

    for o in range(omegaStepsAmount + add):
        for i in range(L):
            sum = sum + R.y[i] * W(i, h, L*h) * np.cos(o * omegaStep * i * h)
        res.append(h/np.pi*sum)
        sum = 0

    frequensyStep = (0.5 + 0.5) / (omegaStepsAmount + add - 1)
    for o in range(omegaStepsAmount + add):
        fres.append(-0.5 + o * frequensyStep)

    newRes = []
    for o in range(omegaStepsAmount + add):
        half = int((omegaStepsAmount + add) / 2)
        if o < half:
            newRes.append(res[o + half])
        elif o >= half:
            newRes.append(res[o - half])
    return pd.DataFrame(
        data=newRes,
        index=fres,
        columns=['y']
    )


def test(omegaStepsAmount):
    add = 1 if omegaStepsAmount % 2 == 0 else 0
    res = []
    fres = []
    frequensyStep = (0.5 + 0.5) / (omegaStepsAmount + add - 1)
    for o in range(omegaStepsAmount + add):
        fres.append(-0.5 + o * frequensyStep)

    n = []
    for i in fres:
        n.append(i*2)
    return pd.DataFrame(
        data=n,
        index=fres,
        columns=['y']
    )


def power_spectral_plot(data1, data2, title):
    fig, axes = plt.subplots(figsize=(17, 5))
    fig.suptitle(title)
    axes.plot(data1, label='non-centered signal', color='cyan', linestyle='--', linewidth=1.4)
    axes.plot(data2, label='centered signal', color='blue', linestyle='--', linewidth=1.4)
    axes.set_xlabel('Frequency, ' + r'$\nu$' + ', Hz')
    axes.set_ylabel('Power Spectral Density, ' + r'$\hat{f}_0(\nu)$' + ', c.u.')
    axes.legend(shadow=True, fancybox=True)
    # axes.legend
    plt.show()


def autocovariance_plot(data1, data2, title):
    fig, axes = plt.subplots(figsize=(15, 5))
    fig.suptitle(title)
    axes.plot(data1, label='non-centered signal', color='cyan')
    axes.plot(data2, label='centered signal', color='blue')
    axes.set_xlabel('Lag, ' + 'u' + '')
    axes.set_ylabel('The Avto-covariance function, ' + r'$B(u)$' + ', c.u.')
    axes.legend(shadow=True, fancybox=True)
    # axes.legend
    plt.show()


def water_level_plot(data, title='Dnister flow rate', freq='week', xlabel='Date, weeks'):
    if freq == 'day':
        return data.plot(
            x='ds',
            y='y',
            xlabel=xlabel,
            ylabel='River level, ' + r'$h$' + ', сm.',
            title=title,
            legend='',
            figsize=(15, 5),
            color='blue'
        )
    else:
        return data.plot(
            # x='ds',
            y='y',
            xlabel=xlabel,
            ylabel='River level, ' + r'$h$' + ', сm.',
            title=title,
            legend='',
            figsize=(15, 5),
            color='blue'
        )


def water_flow_plot(data, title='Dnister flow rate', freq='week', xlabel='Date, weeks'):
    if freq == 'day':
        return data.plot(
            x='ds',
            y='y',
            xlabel=xlabel,
            ylabel='River flow rate, ' + r'$Q$' + r'$, m^3/s$',
            title=title,
            legend='',
            figsize=(15, 5),
            color='blue'
        )
    else:
        return data.plot(
            # x='ds',
            y='y',
            xlabel=xlabel,
            ylabel='River flow rate, ' + r'$Q$' + r'$, m^3/s$',
            title=title,
            legend='',
            figsize=(15, 5),
            color='blue'
        )


def sarimax_gridsearch(ts, pdq, pdqs, maxiter=50):
    ans = []
    for comb in pdq:
        for combs in pdqs:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts,
                                                order=comb,
                                                seasonal_order=combs,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                output = mod.fit(maxiter=maxiter)
                ans.append([comb, combs, output.bic])
                print('SARIMAX {} x {} : BIC Calculated ={}'.format(comb, combs, output.bic))
            except:
                continue

    # Find the parameters with minimal BIC value

    # Convert into dataframe
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'bic'])

    # Sort and return top 5 combinations0
    ans_df = ans_df.sort_values(by=['bic'], ascending=True)[0:5]

    return ans_df
