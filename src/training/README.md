# Анализ ветки 

Данная ветка содержит три основных компонента, предназначенных для тренировки модели глубокого обучения:

- `__init__.py`: инициализирует пакет Python, позволяя импортировать модули внутри пакета.
- `dataset.py`: определяет класс DataLoader, ответственный за подготовку и предоставление данных для обучения.
- `train.py`: реализует основную логику тренировки модели, включая циклы итерации, оценку показателей и сохранение контрольных точек.

Кроме того, имеется важное замечание относительно преобразования типов данных:

> cuFFT поддерживает только степени двойки для размеров массивов в половинной точности (float16). При включённом автоматическом смешанном режиме точности (AMP) тензоры преобразовываются в float16, вызывая ошибки на сетке размером 100×100.<br/>
> Решение: Всегда выполняйте операции FFT в float32 и возвращайтесь обратно к оригинальному типу данных.

Это примечание затрагивает компоненты:

- SpectralConv2d: свёрточный слой спектрального типа.
- SpectralAttention: механизм внимания на основе спектра.
- SpectralLoss: специализированная функция потерь для спектральных операций.
- compute\_metrics: модуль расчёта метрик, зависящих от спектральных характеристик.

⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# Документация для разработчиков

## 1. Подготовка данных 

Класс Dataset предназначен для подготовки и подачи данных для модели. Реализует интерфейс PyTorch Dataset и обеспечивает поддержку функций вроде разделения на тренировочные и тестовые подмножества, а также предварительную обработку данных.

class Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = np.load(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

 ## 2. Процесс обучения 

Основная логика тренировки находится в файле train.py. Здесь реализованы циклы итерации, проверка состояния модели и сохранение чекпоинтов.

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Training complete.")

 ## 3. Решение проблемы с типом данных

Все операции Fast Fourier Transform (FFT) должны проводиться в формате float32, даже если активирован режим автоматического смешанного режима точности (AMP). Таким образом, рекомендуемые шаги включают:

- Выполнение всех FFT-операций в float32.
- Возвращение к оригинальному типу данных (например, float16) после завершения операций.

Например, в классе SpectralConv2d операция выглядит так:

def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
        fft_x = torch.fft.rfftn(x.float(), dim=(2, 3))
        output = torch.irfftn(fft_x * self.weights, s=x.shape[-2:], dim=(2, 3)).type(x.dtype)
    return output

Здесь мы временно отключаем AMP для конкретной части вычислений, выполняя их в float32, а затем возвращаемся к исходному типу данных.
