#!/usr/bin/env python3
"""
tune_weights.py — подбор весов скоринга по ручным выборам из контактного листа.

1. Откройте contact_sheet.html, кликами выберите лучший кадр в нескольких десятках серий
   (чем больше и «разнообразнее» серий, тем лучше) и нажмите «Скачать manual_choices.json».
2. Запустите:
     python tune_weights.py <папка_результатов>/report.json manual_choices.json

Скрипт обучает логистическую регрессию на парах «выбранный кадр — остальные кадры серии»
и печатает: насколько текущие веса совпадают с вашим выбором и предложенный блок
`scoring.weights` для config.yaml. Жёсткие фильтры не меняются.
"""

import argparse
import json
import sys

import numpy as np


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def agreement(persons, choices, weights, keys):
    """Доля серий, где кадр с максимальным score совпал с ручным выбором."""
    w = np.array([weights.get(k, 0.0) for k in keys])
    hit = total = 0
    for p in persons:
        chosen = choices.get(str(p['person_id']))
        if not chosen:
            continue
        scores = [(float(np.dot(w, [f['metrics'].get(k, 0.5) for k in keys])), f['file'])
                  for f in p['frames']]
        total += 1
        hit += max(scores)[1] == chosen
    return hit, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('report')
    ap.add_argument('choices')
    ap.add_argument('--C', type=float, default=1.0,
                    help='регуляризация (меньше — осторожнее при малом числе выборов)')
    args = ap.parse_args()

    report = load(args.report)
    choices = load(args.choices).get('choices', {})
    persons = report['persons']
    keys = list(report['params'].get('weights', {}).keys())
    if not keys:
        sys.exit('В report.json нет params.weights')

    X, y = [], []
    n_series = 0
    for p in persons:
        chosen = choices.get(str(p['person_id']))
        if not chosen or len(p['frames']) < 2:
            continue
        feats = {f['file']: [f['metrics'].get(k, 0.5) for k in keys] for f in p['frames']}
        if chosen not in feats:
            continue
        n_series += 1
        for name, v in feats.items():
            if name == chosen:
                continue
            d = np.array(feats[chosen]) - np.array(v)
            X.append(d); y.append(1)
            X.append(-d); y.append(0)

    if n_series < 5:
        sys.exit(f'Слишком мало ручных выборов ({n_series}). Нужно хотя бы 5, лучше 30+.')

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(fit_intercept=False, C=args.C, max_iter=1000)
    clf.fit(np.array(X), np.array(y))
    coef = np.clip(clf.coef_[0], 0.0, None)
    if coef.sum() < 1e-9:
        sys.exit('Не удалось подобрать веса: все коэффициенты ≤ 0.')
    new_w = {k: float(c / coef.sum()) for k, c in zip(keys, coef)}

    cur_w = report['params']['weights']
    h0, t0 = agreement(persons, choices, cur_w, keys)
    h1, t1 = agreement(persons, choices, new_w, keys)
    print(f'Серий с ручным выбором: {n_series}')
    print(f'Совпадение с текущими весами:    {h0}/{t0}')
    print(f'Совпадение с подобранными весами: {h1}/{t1}  (на тех же данных — оптимистично)')
    if n_series < 30:
        print('Внимание: выборок мало, веса могут быть переобучены; уменьшите --C.')
    print('\nscoring:\n  weights:')
    for k in keys:
        print(f'    {k}: {new_w[k]:.3f}')


if __name__ == '__main__':
    main()
