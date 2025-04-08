## Roadmap

- [x] ver1: запускаем просто mlp на pushcube pick cube
- [x] ver2: учимся формировать правильно контекст на rollout step: тестим на mlp просто взяв -1 срез
- [x] ver3: имплементируем sequential replay buffer - тестим на mlp просто взяв -1 срез 
- [x] ver4: убираем везде срезы и тестим на трансформере акторе, млп критике 
- [x] ver5: тестим на актор и критик раздельные трансформеры
- [] ver6: тестим на актор и критик с общим трансформером и заморозкой

- [ ] тестим на: PickCube-v1, PegInsertionSide-v1, TwoRobotStackCube-v1, TriFingerRotateCubeLevel0-v1,  PokeCube-v1, PickSingleYBC 
- [ ] аналогично пунктам 1-7 скрещиваем трансформер с PPO 
- [ ] начинаем переписывать под картинки SAC
