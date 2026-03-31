# Dr. Splat 실험 텍스트 쿼리 목록

- **씬**: ramen (LeRF-OVS)
- **GT 소스**: Ref-lerf (FudanCVL/Ref-Lerf, HuggingFace)
- **테스트 프레임**: 7개 (frame_00006, 00024, 00060, 00065, 00081, 00119, 00128)
- **총 쿼리 수**: 68개 (Level 0~5)

---

## Level 0: Base Retrieval (13개)

GT 카테고리 단어 그대로. 정량 평가 가능 (GT mask 존재).

| # | 쿼리 | 대상 객체 | GT 프레임 수 |
|---|---|---|---|
| 1 | bowl | 라멘 그릇 | 7/7 |
| 2 | chopsticks | 젓가락 | 7/7 |
| 3 | corn | 옥수수 | 7/7 |
| 4 | egg | 계란 (단수) | 7/7 |
| 5 | eggs | 계란 (복수) | 7/7 |
| 6 | glass of water | 물잔 | 5/7 |
| 7 | hand | 손 | 1/7 |
| 8 | kamaboko | 나루토/카마보코 | 7/7 |
| 9 | nori | 김 | 7/7 |
| 10 | plate | 접시 | 5/7 |
| 11 | sake cup | 사케잔 | 6/7 |
| 12 | spoon | 숟가락 | 3/7 |
| 13 | wavy noodles | 라멘 면 | 7/7 |

---

## Level 1: 단어/속성 변화 (13개)

GT 카테고리에 색상/재질/외형 속성을 추가. 동일 GT mask로 정량 평가 가능.

| # | 쿼리 | 대응 GT 카테고리 | 추가 속성 |
|---|---|---|---|
| 1 | yellow bowl | bowl | 색상 |
| 2 | wooden chopsticks | chopsticks | 재질 |
| 3 | yellow corn | corn | 색상 |
| 4 | halved egg | egg | 외형 |
| 5 | two eggs | eggs | 수량 |
| 6 | water glass | glass of water | 어순 변경 |
| 7 | person hand | hand | 소유자 |
| 8 | pink kamaboko | kamaboko | 색상 |
| 9 | dark nori | nori | 색상 |
| 10 | metal plate | plate | 재질 |
| 11 | metal cup | sake cup | 재질 |
| 12 | black spoon | spoon | 색상 |
| 13 | curly noodles | wavy noodles | 외형 |

---

## Level 2: 세부 파트 (12개)

객체의 일부분만 지칭. GT mask 없음, 정성 평가.

| # | 쿼리 | 대응 객체 | 지칭 부위 |
|---|---|---|---|
| 1 | bowl rim | bowl | 그릇 테두리 |
| 2 | chopstick tip | chopsticks | 젓가락 끝 |
| 3 | egg yolk | egg | 노른자 |
| 4 | egg white | egg | 흰자 |
| 5 | glass rim | glass of water | 잔 테두리 |
| 6 | fingertips | hand | 손가락 끝 |
| 7 | kamaboko swirl | kamaboko | 소용돌이 무늬 |
| 8 | nori edge | nori | 김 가장자리 |
| 9 | plate rim | plate | 접시 테두리 |
| 10 | cup rim | sake cup | 잔 테두리 |
| 11 | spoon handle | spoon | 손잡이 |
| 12 | noodle strand | wavy noodles | 면 가닥 |
| 13 | water | glass of water | 물 (내용물만) |
| 14 | bottle | sake cup 근처 | 병 (용기 분리) |

---

## Level 3: 같은 종류 구분 (4개)

동일 카테고리 내 개별 인스턴스 구분. GT mask 없음, 정성 평가.

| # | 쿼리 | 테스트 의도 |
|---|---|---|
| 1 | left chopstick | 좌/우 젓가락 구분 |
| 2 | left egg | 좌/우 계란 구분 |
| 3 | both eggs | 복수 인스턴스 동시 선택 |
| 4 | right nori | 좌/우 김 구분 |

---

## Level 4: 관계 (13개)

공간 관계(on, beside, near, above, below, behind, under)로 객체 지칭. GT mask 없음, 정성 평가.

| # | 쿼리 | 관계 유형 | 대응 GT 카테고리 |
|---|---|---|---|
| 1 | bowl on plate | on | bowl |
| 2 | chopsticks beside bowl | beside | chopsticks |
| 3 | corn near egg | near | corn |
| 4 | egg above noodles | above | egg |
| 5 | eggs above noodles | above | eggs |
| 6 | glass beside bowl | beside | glass of water |
| 7 | hand behind bowl | behind | hand |
| 8 | kamaboko below eggs | below | kamaboko |
| 9 | nori beside eggs | beside | nori |
| 10 | plate under bowl | under | plate |
| 11 | cup near chopsticks | near | sake cup |
| 12 | spoon behind bowl | behind | spoon |
| 13 | noodles below eggs | below | wavy noodles |

---

## 쿼리 레벨별 요약

| 레벨 | 설명 | 개수 | 난이도 | 정량 평가 |
|---|---|---|---|---|
| Level 0 | Base Retrieval (GT 카테고리 단어) | 13 | 가장 쉬움 | O (GT mask) |
| Level 1 | 단어/속성 변화 | 13 | 쉬움 | O (GT mask 재활용) |
| Level 2 | 세부 파트 | 14 | 보통 | X (정성) |
| Level 3 | 같은 종류 구분 | 4 | 어려움 | X (정성) |
| Level 4 | 공간 관계 | 13 | 어려움 | X (정성) |
| **합계** | | **57** | | |

---

## Level 5: 설명문 (13개)

Ref-lerf JSON의 `sentence` 필드. 객체를 직접 명명하지 않고 외형/위치/기능으로 서술. GT mask로 정량 평가 가능.

| # | 쿼리 | 대응 GT 카테고리 |
|---|---|---|
| 1 | An object containing noodles and various toppings on a table with sloping edges. | bowl |
| 2 | A pair of utensils for holding food next to a yellow bowl. | chopsticks |
| 3 | Corn kernels placed in a bowl, blended with the ingredients, floating on the surface. | corn |
| 4 | The half of the egg next to the seaweed with the white evenly wrapped around the orange yolk. | egg |
| 5 | Slices cut evenly in half down the center, above the ramen noodles. | eggs |
| 6 | A container with water resting on wooden tabletop and near wall. | glass of water |
| 7 | Presented in a light grip and displaying the natural texture of the knuckles is a hand. | hand |
| 8 | Smooth slices with a delicate spiral pattern featuring a swirl pattern. | kamaboko |
| 9 | Long, dark green seaweed flakes placed against the side of the bowl. | nori |
| 10 | A metal utensil that is convenient for holding cutlery, located on a wooden table and below the yellow bowl. | plate |
| 11 | A drinking utensils with a smooth surface near a sake bottle. | sake cup |
| 12 | The long one on the tray. | spoon |
| 13 | Slender strips of food in the center of the bowl, mixed with toppings. | wavy noodles |
