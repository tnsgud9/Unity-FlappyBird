using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Q-Learning 강화학습 알고리즘
/// 
/// [Q-Learning이란?]
/// 경험을 통해 최적의 행동을 학습하는 알고리즘입니다.
/// 
/// 예를 들어, Flappy Bird에서:
/// - 새가 점프를 했더니 파이프를 통과했다 → "점프가 좋은 행동"구나
/// - 새가 가만히 있었더니 바닥에 부딪혔다 → "가만히 있으면 안 되는 행동"구나!
/// 
/// 이런 경험을 쌰아가며 최적의 행동을 학습합니다.
/// 
/// [핵심 개념 4가지]
/// 1. State (상태): 현재 상황을 나타내는 정보
///    - 예: 새의 Y 위치, 파이프까지의 거리
/// 
/// 2. Action (행동): 선택 가능한 동작
///    - 예: 0 = 대기 (가만히 있기), 1 = 점프 (위로 이동)
/// 
/// 3. Reward (보상): 행동의 결과로 받는 점수
///    - 양수 = 좋은 행동, 음수 = 나쁜 행동
///    - 예: 파이프 통과 = +1.0, 충돌 = -1.0
/// 
/// 4. Q-Value (Q값): 특정 상태에서 특정 행동의 "가치"
///    - Q값이 높을수록 그 행동이 좋은 선택
///    - Q-Table: 모든 상태-행동 쌍의 Q값을 저장한 표
/// 
/// [Q-Learning 공식]
/// Q(s,a) ← Q(s,a) + α × [R + γ × max(Q(s',a')) - Q(s,a)]
/// 
/// 공식 해석:
/// - s: 현재 상태 (state)
/// - a: 선택한 행동 (action)
/// - R: 받은 보상 (reward)
/// - s': 다음 상태 (next state)
/// - max(Q(s',a')): 다음 상태에서 가장 좋은 행동의 Q값
/// 
/// [하이퍼파라미터]
/// α (learningRate, 학습률): 새로운 정보를 얼마나 반영할지
///    - 높을수록 빠르게 학습, 불안정할 수 있음
///    - 낮을수록 천천히 학습, 안정적임
/// 
/// γ (discountFactor, 할인율): 미래 보상을 얼마나 중요하게 볼지
///    - 높을수록 미래를 중시 (장기적 사고)
///    - 낮을수록 현재를 중시 (단기적 사고)
/// 
/// ε (epsilon, 탐색률): 무작위 행동을 시도할 확률
///    - 높을수록 많이 탐색 (새로운 시도)
///    - 낮을수록 많이 활용 (학습한 지식 사용)
/// </summary>
public class QLearning
{
    // ========================================
    // Q-Table: 각 상태에서 각 행동의 가치를 저장
    // Key = (stateY, distY, pipeDistX)
    // Value = [대기의 Q값, 점프의 Q값]
    // ========================================
    private Dictionary<(int, int, int), float[]> _qTable;
    
    // ========================================
    // 하이퍼파라미터
    // ========================================
    private float _learningRate;     // 학습률 (α): 0~1
    private float _discountFactor;   // 할인율 (γ): 0~1
    private float _epsilon;          // 탐색률 (ε): 0~1
    
    private int _actionCount = 2;    // 행동 개수 (0=대기, 1=점프)
    
    // ========================================
    // [고급] Best Q-Table 저장용
    // 유전 알고리즘처럼 좋은 정책을 보존
    // ========================================
    private Dictionary<(int, int, int), float[]> _bestQTable;
    private float _bestScore;
    
    /// <summary>
    /// Q-Learning 초기화
    /// </summary>
    /// <param name="learningRate">학습률 (기본값 0.1)</param>
    /// <param name="discountFactor">할인율 (기본값 0.9)</param>
    /// <param name="epsilon">탐색률 (기본값 1.0)</param>
    public QLearning(float learningRate = 0.1f, float discountFactor = 0.9f, float epsilon = 1f)
    {
        // Q-Table 생성
        _qTable = new Dictionary<(int, int, int), float[]>();
        _bestQTable = null;
        
        // 하이퍼파라미터 설정
        _learningRate = learningRate;
        _discountFactor = discountFactor;
        _epsilon = epsilon;
        _bestScore = 0f;
    }
    
    // ========================================
    // 핵심 메서드: 행동 선택
    // ========================================
    
    /// <summary>
    /// 현재 상태에서 행동 선택 (ε-greedy 정책)
    /// 
    /// [ε-greedy 정책이란?]
    /// ε 확률로 무작위 행동 (탐색)
    /// (1-ε) 확률로 Q값이 높은 행동 (활용)
    /// 
    /// 이 방식으로 탐색과 활용의 균형을 맞춥니다.
    /// </summary>
    public int GetAction((int, int, int) state)
    {
        // 새로운 상태면 Q-Value 초기화
        // Q값이 0.01로 초기화되어 어떤 행동도 나쁘지 않음
        if (!_qTable.ContainsKey(state))
            _qTable[state] = new float[] { 0.01f, 0.01f };
        
        // ε 확률로 무작위 행동 선택 (탐색)
        // Random.value: 0~1 사이의 랜덤 값
        // _epsilon보다 작으면 무작위 행동
        if (Random.value < _epsilon)
            return Random.Range(0, _actionCount);
        
        // Q-Value가 비슷하면 대기 선호 (안전한 선택)
        // Q값 차이가 0.01 미만이면 아직 학습이 덜 됨
        float qDiff = _qTable[state][0] - _qTable[state][1];
        if (Mathf.Abs(qDiff) < 0.01f)
            return 0;  // 대기 선택
        
        // Q-Value가 더 높은 행동 선택 (활용)
        // 대기 Q값 >= 점프 Q값 → 대기 (0)
        // 대기 Q값 < 점프 Q값 → 점프 (1)
        return _qTable[state][0] >= _qTable[state][1] ? 0 : 1;
    }
    
    // ========================================
    // 핵심 메서드: Q-Value 갱신 (학습)
    // ========================================
    
    /// <summary>
    /// Q-Value 갱신 (학습)
    /// 
    /// [학습 과정]
    /// 1. 현재 상태에서 행동을 선택함
    /// 2. 행동의 결과로 보상을 받음
    /// 3. 다음 상태로 이동
    /// 4. Q-Value를 갱신하여 경험을 저장
    /// 
    /// [공식]
    /// 새로운 Q값 = 기존 Q값 + 학습률 × (보상 + 할인율 × 미래 최대 Q값 - 기존 Q값)
    /// </summary>
    public void Update((int, int, int) state, int action, float reward, (int, int, int) nextState)
    {
        // 현재 상태가 Q-Table에 없으면 추가
        if (!_qTable.ContainsKey(state))
            _qTable[state] = new float[] { 0.01f, 0.01f };
        
        // 다음 상태가 Q-Table에 없으면 추가
        if (!_qTable.ContainsKey(nextState))
            _qTable[nextState] = new float[] { 0.01f, 0.01f };
        
        // 다음 상태에서 가능한 최대 Q값 찾기
        // 이 값은 "미래에 얼마나 좋을 수 있는지"를 나타냄
        float maxNextQ = Mathf.Max(_qTable[nextState]);
        
        // Q-Learning 공식 적용
        // 새로운 Q값 = 기존 Q값 + 학습률 × (보상 + 할인율×미래보상 - 기존 Q값)
        float oldValue = _qTable[state][action];
        float futureReward = reward + _discountFactor * maxNextQ;
        float tdError = futureReward - oldValue;
        _qTable[state][action] = oldValue + _learningRate * tdError;
    }
    
    // ========================================
    // [고급] Q-Table 관리 메서드
    // ========================================
    
    /// <summary>
    /// [고급] 최고 성과 Q-Table 저장
    /// 
    /// 유전 알고리즘처럼 좋은 정책을 보존합니다.
    /// 최고 점수를 기록하면 현재 Q-Table을 저장합니다.
    /// 나중에 학습이 잘못되면 이전 좋은 정책으로 복원할 수 있습니다.
    /// </summary>
    public void SaveBestQTable(float score)
    {
        // 새로운 최고 점수면 저장
        if (score > _bestScore)
        {
            _bestScore = score;
            
            // 현재 Q-Table을 깊은 복사로 저장
            _bestQTable = new Dictionary<(int, int, int), float[]>();
            foreach (var kvp in _qTable)
            {
                // 배열을 복사 (참조가 아닌 값 복사)
                _bestQTable[kvp.Key] = (float[])kvp.Value.Clone();
            }
        }
    }
    
    /// <summary>
    /// [고급] 최고 성과 Q-Table 복원
    /// 
    /// 학습이 잘못된 방향으로 가면 저장된 최고 정책을 복원합니다.
    /// 복원 시 Epsilon을 약간 높여 재탐색을 유도합니다.
    /// </summary>
    public void RestoreBestQTable()
    {
        // 저장된 Best Q-Table이 없으면 아무것도 하지 않음
        if (_bestQTable == null) return;
        
        // Best Q-Table로 현재 Q-Table 교체
        _qTable = new Dictionary<(int, int, int), float[]>();
        foreach (var kvp in _bestQTable)
        {
            _qTable[kvp.Key] = (float[])kvp.Value.Clone();
        }
    }
    
    // ========================================
    // Epsilon 관리
    // ========================================
    
    /// <summary>
    /// 탐색률(Epsilon) 설정
    /// </summary>
    public void SetEpsilon(float value) => _epsilon = Mathf.Clamp01(value);
    
    /// <summary>
    /// 현재 탐색률(Epsilon) 반환
    /// </summary>
    public float GetEpsilon() => _epsilon;
}
