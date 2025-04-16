## Roadmap

- [x] ver1: запускаем просто mlp на pushcube pick cube
- [x] ver2: учимся формировать правильно контекст на rollout step: тестим на mlp просто взяв -1 срез
- [x] ver3: имплементируем sequential replay buffer - тестим на mlp просто взяв -1 срез 
- [x] ver4: убираем везде срезы и тестим на трансформере акторе, млп критике 
- [x] ver5: тестим на актор и критик раздельные трансформеры
<<<<<<< HEAD
- [] ver6: тестим на актор и критик с общим трансформером и заморозкой

- [ ] тестим на: PickCube-v1, PegInsertionSide-v1, TwoRobotStackCube-v1, TriFingerRotateCubeLevel0-v1,  PokeCube-v1, PickSingleYBC 
=======
- [ ] тестим ver5 на: **Простые среды:** PickCube, Push-T, StackSube | **Сложные среды:** TriFingerRotateCubeLevel0-v1, TwoRobotStackCube-v1
- [ ] тестим ver5 на: **Потом:** PegInsertionSide-v1, PokeCube-v1, PickSingleYBC
- [ ] ver6: тестим на актор и критик с общим трансформером и заморозкой

>>>>>>> 821f1e9ee020130a41f45d79060ca04a28b3c8bb
- [ ] аналогично пунктам 1-7 скрещиваем трансформер с PPO 
- [ ] начинаем переписывать под картинки SAC
