# ServerBalancer

Прототип балансировщика нагрузки на основе теории массового обслуживания: оптимальное распределение трафика между серверами через LP/NLP-оптимизацию с моделью M/M/1.

## Математическая модель

Каждый сервер моделируется как очередь M/M/1. Среднее время пребывания запроса:

```
W_i = 1 / (mu_i - lambda_i)
```

Три постановки задачи оптимизации долей трафика x_i (sum x_i = 1, lambda_i = x_i * lambda_total):

- **LP balanced** — LP минимизирует максимальную загрузку rho_i (min-max fairness)
- **LP proxy** — LP минимизирует линейный суррогат времени отклика 1/(mu_i - lambda_i) (аппроксимация)
- **NLP SLSQP** — SciPy SLSQP минимизирует истинное W_avg = sum(x_i * W_i) напрямую

## Структура проекта

```
server_balancer/
├── main.py              # CLI: анализ, запуск API-сервера
├── config.yaml          # параметры по умолчанию (lambda, mu серверов, rho_max)
├── requirements.txt     # зависимости Python
├── Dockerfile           # образ для запуска API
├── api/
│   └── app.py           # FastAPI приложение (эндпоинты /balance, /compare, /health и др.)
├── core/
│   ├── queue_math.py    # формулы M/M/1: метрики, среднее время отклика
│   ├── lp_optimizer.py  # три постановки LP/NLP (PuLP + SciPy SLSQP)
│   ├── simulation.py    # SimPy дискретно-событийная симуляция для валидации
│   └── plots.py         # вспомогательная визуализация
└── tests/
    ├── test_queue_math.py     # тесты аналитических формул M/M/1
    ├── test_lp_optimizer.py   # тесты трёх постановок оптимизатора
    └── test_api.py            # интеграционные тесты FastAPI
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

### CLI (анализ)

```bash
# параметры из config.yaml
python main.py

# переопределить суммарный поток
python main.py --lambda-total 200

# с SimPy-валидацией
python main.py --sim
```

### API сервер

```bash
python main.py --serve
```

Интерактивная документация: http://localhost:8000/docs

### Примеры запросов к API

```bash
# Оптимизация NLP (минимизация истинного W)
curl -X POST http://localhost:8000/balance \
  -H "Content-Type: application/json" \
  -d '{"lambda_total": 150, "servers": [{"id":0,"mu":100},{"id":1,"mu":80},{"id":2,"mu":60}], "formulation": "nlp_true"}'

# Оптимизация LP balanced (min-max загрузка)
curl -X POST http://localhost:8000/balance \
  -H "Content-Type: application/json" \
  -d '{"lambda_total": 150, "servers": [{"id":0,"mu":100},{"id":1,"mu":80},{"id":2,"mu":60}], "formulation": "balanced"}'

# Сравнение всех трёх постановок одним запросом
curl "http://localhost:8000/compare?lambda_total=150&mu1=100&mu2=80&mu3=60"

# Проверка живости
curl http://localhost:8000/health
```

### Docker

```bash
docker build -t server-balancer .
docker run -p 8000:8000 server-balancer
```

## Тесты

```bash
pytest tests/ -v --cov=core --cov=api --cov-report=term-missing
```

## Конфигурация

Файл `config.yaml`:

| Поле | Тип | Описание |
|------|-----|----------|
| `lambda_total` | float | Суммарный поток запросов, req/s |
| `rho_max` | float | Максимально допустимая загрузка сервера (0 < rho_max < 1) |
| `servers[].id` | int | Идентификатор сервера |
| `servers[].mu` | float | Интенсивность обслуживания, req/s |
| `simulation_duration` | float | Длительность SimPy-симуляции, секунд модельного времени |

## Результаты (вариант: lambda=150, mu=[100, 80, 60])

При равных долях x_i = 1/3 каждый сервер получает lambda_i = 50 req/s:
W_0 = 1/(100-50) = 20 мс, W_1 = 1/(80-50) = 33 мс, W_2 = 1/(60-50) = 100 мс, W_avg = 51 мс.

NLP перераспределяет трафик в пользу быстрых серверов, значительно снижая W_avg.

| Метод | W_avg, мс | Описание |
|-------|-----------|----------|
| Равные доли (baseline) | 51.111 | x = [1/3, 1/3, 1/3], без оптимизации |
| LP balanced | ~14–16 | Минимизирует max rho_i, трафик пропорционален mu_i |
| LP proxy | ~9–11 | Линейный суррогат W, лучше учитывает производительность |
| NLP SLSQP | ~7–9 | Минимизирует истинный W_avg, наилучший результат |
