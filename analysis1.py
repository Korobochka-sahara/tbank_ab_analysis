from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

SOURCE = Path(__file__).resolve().parent/'dannye-dlia-keisa-po-pa-cebbefc3-5b29-4f88-b6d4-3336c88d743f.xlsx'
OUT = Path(__file__).resolve().parent
RESULTS = OUT / 'results'
PLOTS = RESULTS / 'plots'
RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

u=pd.read_excel(SOURCE, sheet_name='Пользователи')
p=pd.read_excel(SOURCE, sheet_name='Платежи')
log = []

log.append(('Исходные пользователи', len(u)))
log.append(('Исходные попытки', len(p)))
log.append(('Пользователи с несуществующей группой', (~u.group.isin(['A','B'])).sum()))
log.append(('Пользователи с аномальным возрастом (<18,>100)', (~u.age.between(18,100)).sum()))
log.append(('Платежи без user_id', p.user_id.isna().sum()))
log.append(('Платежи без payment_id', p.payment_id.isna().sum()))
log.append(('Пользователи без user_id',u.user_id.isna().sum()))
log.append(('Дубли user_id в таблице пользователей',u.user_id.duplicated().sum()))
log.append(('Дубли payment_id',p.payment_id.duplicated().sum()))

check = p.merge(
    u[['user_id', 'group']],
    on = 'user_id',
    how = 'left',
    suffixes=('_payment', '_user')
)
mismatch = (
    check.group_payment.notna()
    & check.group_user.notna()
    & (check.group_payment != check.group_user)
)
log.append(('Несовпадения группы пользователя в таблице платежей', mismatch.sum()))
log.append(('Платежи пользователей, отсутствующих в таблице пользователей', (p.user_id.notna() & ~p.user_id.isin(u.user_id)).sum()))

start = pd.Timestamp('2024-04-01')
end = pd.Timestamp('2024-05-01')
invalid_dates = ~p.step1_opened.between(start, end, inclusive='left')
log.append(('Платежи с датой открытия вне периода теста',invalid_dates.sum()))

steps = [
    'step1_opened',
    'step2_entered',
    'step3_confirmed',
    'step4_success'
]
invalid_steps = pd.Series(False, index=p.index)
for i in range(1, len(steps)):
    prev = p[steps[i-1]]
    curr = p[steps[i]]

    invalid_steps |= (
        curr.notna()
        & (prev.isna() | (curr < prev))
    )
log.append(('Платежи с нарушением последовательности шагов',invalid_steps.sum()))

log_df = pd.DataFrame(log, columns=['Проверка', 'Количество'])
print(log_df.to_string(index=False))
print('Размер групп:')
print(u.group.value_counts(dropna=False))
users_before = len(u)
payments_before = len(p)

log_df.to_csv(
    RESULTS / 'cleaning_log.csv',
    index=False,
    encoding='utf-8-sig'
)

# ОЧИСТКА ДАННЫХ

# Возраст нам не важен, заменим на nan
u.loc[~u.age.between(18, 100), 'age'] = np.nan
# Удаляем пользователей с недостоверными группами
u = u[u.group.isin(['A', 'B'])].copy()
# удаляем платежи без user_id
p = p.dropna(subset=['user_id']).copy()
# Объединяем платежи с пользователями
p = p.merge(
    u[['user_id', 'group', 'device_type', 'city', 'age']],
    on='user_id',
    how='left',
    suffixes=('_payment', '_user'),
    validate='many_to_one'
)
# Удаляем платежи, для которых нет валидного пользователя
p = p[p.group_user.notna()].copy()
# Используем группу из таблицы пользователей
p['group'] = p.group_user

# Проверяем результаты очистки
print('Пользователей удалено:', users_before - len(u))
print('Платежей удалено:', payments_before - len(p))
print('Пользователей осталось:', len(u))
print('Платежей осталось:', len(p))

for i, step in enumerate(steps, 1):
    p[f's{i}'] = p[step].notna().astype(int)
funnel = p.groupby('group')[['s1', 's2', 's3', 's4']].sum()
print('\nВоронка')
print(funnel)
conversion = funnel.div(funnel['s1'], axis=0) * 100
print('\nНакопительная конверсия')
print(conversion.round(2))

step_conversion = pd.DataFrame(index=funnel.index)
step_conversion['1 -> 2'] = funnel['s2'] / funnel['s1'] * 100
step_conversion['2 -> 3'] = funnel['s3'] / funnel['s2'] * 100
step_conversion['3 -> 4'] = funnel['s4'] / funnel['s3'] * 100
print('\nПошаговая конверсия')
print(step_conversion.round(2))

cr_a = conversion.loc['A', 's4']
cr_b = conversion.loc['B', 's4']
absolute_lift = cr_b - cr_a
relative_lift = (cr_b / cr_a - 1) * 100
print(f'\nКонверсия A: {cr_a:.2f}%')
print(f'Конверсия B: {cr_b:.2f}%')
print(f'Абсолютный прирост: {absolute_lift:.2f} процентных пунктов')
print(f'Относительный прирост: {relative_lift:.2f}%')

user_metrics = p.groupby('group').agg(
    active_users=('user_id', 'nunique'),
    attempts=('payment_id', 'size')
)
user_metrics['attempts_per_user'] = (
    user_metrics['attempts'] / user_metrics['active_users']
)
print('\nАктивность пользователей')

print(user_metrics.round(2))
user_success = p.groupby(['group', 'user_id'])['s4'].max()
user_conversion = (user_success.groupby('group').mean() * 100)
print('\nПользовательская конверсия')
print(user_conversion.round(2))

dropoff = 100 - step_conversion
print('\nПотери на этапах воронки (%)')
print(dropoff.round(2))

cleaning_summary = pd.DataFrame({
    'Метрика': [
        'Пользователей до очистки',
        'Пользователей после очистки',
        'Удалено пользователей',
        'Платежей до очистки',
        'Платежей после очистки',
        'Удалено платежей'
    ],
    'Количество': [
        users_before,
        len(u),
        users_before - len(u),
        payments_before,
        len(p),
        payments_before - len(p)
    ]
})
cleaning_summary.to_csv(
    RESULTS / 'cleaning_summary.csv',
    index=False,
    encoding='utf-8-sig'
)

# ГРАФИКИ

# Воронка оплаты по группам
fig, ax = plt.subplots(figsize=(9, 5))
labels = ['Открытие', 'Реквизиты', 'Подтверждение', 'Успех']

for group in ['A', 'B']:
    ax.plot(
        labels,
        conversion.loc[group].values,
        marker='o',
        linewidth=2,
        label=f'Группа {group}'
    )
ax.set_title('Воронка оплаты ЖКХ')
ax.set_ylabel('Конверсия от открытия формы, %')
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(PLOTS / 'funnel.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Сравнение основной конверсии
fig, ax = plt.subplots(figsize=(6, 5))

groups = ['A', 'B']
values = [cr_a, cr_b]

bars = ax.bar(groups, values)

ax.bar_label(bars, fmt='%.2f%%', padding=3)

ax.set_title('Конверсия в успешную оплату')
ax.set_ylabel('Конверсия, %')
ax.set_ylim(0, 80)

plt.tight_layout()
fig.savefig(PLOTS / 'conversion.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Потери на каждом этапе

ax = dropoff.T.plot(
    kind='bar',
    figsize=(9, 5)
)

plt.title('Потери пользователей на этапах оплаты')
plt.ylabel('Доля потерь, %')
plt.xlabel('Переход между шагами')
plt.xticks(rotation=0)
plt.legend(title='Группа')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
ax.figure.savefig(
    PLOTS / 'dropoff.png',
    dpi=200,
    bbox_inches='tight'
)
plt.close(ax.figure)

# БУСТРАП АНАЛИЗ 
user = p.groupby(['user_id','group']).agg(
    attemps=('s4', 'size'), 
    successes = ('s4','sum')
).reset_index()

a = user[user.group == 'A']
b = user[user.group == 'B']

arr_a = a[['successes', 'attemps']].to_numpy()
arr_b = b[['successes', 'attemps']].to_numpy()

rng = np.random.default_rng(42)

N = 10000
boot = np.empty(N)
for i in range(N):
    sample_a = arr_a[rng.integers(0, len(arr_a), len(arr_a))]
    sample_b = arr_b[rng.integers(0, len(arr_b), len(arr_b))]

    conv_a = sample_a[:, 0].sum() / sample_a[:, 1].sum()
    conv_b = sample_b[:, 0].sum() / sample_b[:, 1].sum()

    boot[i] = conv_b - conv_a

ci = np.quantile(boot, [0.025, 0.975])
print('\nДоверительный интервал разницы конверсий бустрапом')
print(ci * 100)

# ПЕРЕСТАНОВОЧНЫЙ ТЕСТ (меняем местами значения из A и B, и проверяем, насколько изменится конверсия)

all_users = np.vstack([arr_a, arr_b])
n_a = len(arr_a)

N = 10000
permutations = np.empty(N)

for i in range(N):
    shuffled = rng.permutation(len(all_users))

    sample_a = all_users[shuffled[:n_a]]
    sample_b = all_users[shuffled[n_a:]]

    conv_a = sample_a[:, 0].sum() / sample_a[:, 1].sum()
    conv_b = sample_b[:, 0].sum() / sample_b[:, 1].sum()

    permutations[i] = conv_b - conv_a

# Наблюдаемая разница
observed_diff = cr_b / 100 - cr_a / 100
# сколько перестановок дали разницу не меньше наблюдаемой по модулю
extreme = np.sum(
    np.abs(permutations) >= abs(observed_diff)
)
# вычисляем p-value
p_value = (extreme + 1) / (N + 1)
print(f'\np-value: {p_value:.6f}')

# Вычисляем SRM для p-value
counts = u.group.value_counts().reindex(['A', 'B'])
stat, srm_p = chisquare(counts)
print(f'SRM p-value: {srm_p:.6f}')

# ИТОГОВЫЕ МЕТРИКИ ДЛЯ ОТЧЁТА

metrics_summary = pd.DataFrame({
    'Метрика': [
        'Пользователей A',
        'Пользователей B',
        'Активных пользователей A',
        'Активных пользователей B',
        'Попыток A',
        'Попыток B',
        'Успешных оплат A',
        'Успешных оплат B',
        'Конверсия попыток A (%)',
        'Конверсия попыток B (%)',
        'Пользовательская конверсия A (%)',
        'Пользовательская конверсия B (%)',
        'Абсолютный прирост (п.п.)',
        'Относительный прирост (%)',
        'Нижняя граница 95% CI (п.п.)',
        'Верхняя граница 95% CI (п.п.)',
        'Permutation p-value',
        'SRM p-value'
    ],
    'Значение': [
        (u.group == 'A').sum(),
        (u.group == 'B').sum(),
        user_metrics.loc['A', 'active_users'],
        user_metrics.loc['B', 'active_users'],
        funnel.loc['A', 's1'],
        funnel.loc['B', 's1'],
        funnel.loc['A', 's4'],
        funnel.loc['B', 's4'],
        cr_a,
        cr_b,
        user_conversion.loc['A'],
        user_conversion.loc['B'],
        absolute_lift,
        relative_lift,
        ci[0] * 100,
        ci[1] * 100,
        p_value,
        srm_p
    ]
})

metrics_summary.to_csv(
    RESULTS / 'metrics_summary.csv',
    index=False,
    encoding='utf-8-sig'
)

print('\nИтоговые метрики:')
print(metrics_summary.to_string(index=False))