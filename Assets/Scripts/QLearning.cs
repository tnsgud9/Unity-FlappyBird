using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Q-Learning 강화학습 알고리즘
/// 
/// [Q-Learning이란?]
/// 경험을 통해 최적의 행동을 학습하는 알고리즘입니다.
/// 
/// [핵심 개념]
/// - State (상태): 현재 상황 (예: 새의 위치, 파이프까지의 거리)
/// - Action (행동): 선택 가능한 동작 (0=대기, 1=점프)
/// - Reward (보상): 행동의 결과 (양수=좋음, 음수=나쁨)
/// - Q-Value: 특정 상태에서 특정 행동의 가치
/// 
/// [Q-Learning 공식]
/// Q(s,a) ← Q(s,a) + α × [R + γ × max(Q(s',a')) - Q(s,a)]
/// 
/// - α (alpha): 학습률 - 새로운 정보를 얼마나 반영할지
/// - γ (gamma): 할인율 - 미래 보상을 얼마나 중요하게 볼지
/// - ε (epsilon): 탐색률 - 새로운 행동을 무작위로 시도할 확률
/// </summary>
public class QLearning
{
    // ========================================
    // Q-Table: 각 상태에서 각 행동의 가치를 저장
    // Key = (distY, pipeDistX), Value = [대기가치, 점프가치]
    // ========================================
    private Dictionary<(int, int), float[]> _qTable;
    
    // ========================================
    // 하이퍼파라미터
    // ========================================
    private float _learningRate;     // 학습률 (α): 0~1
    private float _discountFactor;   // 할인율 (γ): 0~1
    private float _epsilon;          // 탐색률 (ε): 0~1
    
    private int _actionCount = 2;    // 행동 개수 (0=대기, 1=점프)
    
    // ========================================
    // [고급] Best Q-Table 저장용
    // ========================================
    private Dictionary<(int, int), float[]> _bestQTable;
    private float _bestScore;
    
    /// <summary>
    /// Q-Learning 초기화
    /// </summary>
    /// <param name="learningRate">학습률: 높을수록 새 정보를 빠르게 반영</param>
    /// <param name="discountFactor">할인율: 높을수록 미래 보상을 중시</param>
    /// <param name="epsilon">탐색률: 높을수록 무작위 행동을 많이 시도</param>
    public QLearning(float learningRate = 0.1f, float discountFactor = 0.9f, float epsilon = 1f)
    {
        _qTable = new Dictionary<(int, int), float[]>();
        _bestQTable = null;
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
    /// [동작 방식]
    /// 1. ε 확률로 무작위 행동 선택 (탐색)
    /// 2. 그 외에는 Q-Value가 가장 높은 행동 선택 (활용)
    /// </summary>
    public int GetAction((int, int) state)
    {
        // 새로운 상태면 Q-Value 초기화
        if (!_qTable.ContainsKey(state))
            _qTable[state] = new float[] { 0.01f, 0.01f };
        
        // ε 확률로 무작위 행동 (탐색)
        if (Random.value < _epsilon)
            return Random.Range(0, _actionCount);
        
        // Q-Value가 비슷하면 대기 선호 (안전한 선택)
        float qDiff = _qTable[state][0] - _qTable[state][1];
        if (Mathf.Abs(qDiff) < 0.01f)
            return 0;
        
        // Q-Value가 더 높은 행동 선택 (활용)
        return _qTable[state][0] >= _qTable[state][1] ? 0 : 1;
    }
    
    // ========================================
    // 핵심 메서드: Q-Value 갱신
    // ========================================
    
    /// <summary>
    /// Q-Value 갱신 (학습)
    /// 
    /// [Q-Learning 공식]
    /// Q(s,a) ← Q(s,a) + α × [R + γ × max(Q(s',a')) - Q(s,a)]
    /// 
    /// 현재 상태에서 행동한 결과, 얻은 보상을 바탕으로 Q-Value 업데이트
    /// </summary>
    public void Update((int, int) state, int action, float reward, (int, int) nextState)
    {
        // 상태가 없으면 초기화
        if (!_qTable.ContainsKey(state))
            _qTable[state] = new float[] { 0.01f, 0.01f };
        if (!_qTable.ContainsKey(nextState))
            _qTable[nextState] = new float[] { 0.01f, 0.01f };
        
        // 다음 상태의 최대 Q-Value
        float maxNextQ = Mathf.Max(_qTable[nextState]);
        
        // Q-Learning 공식 적용
        _qTable[state][action] += _learningRate * (reward + _discountFactor * maxNextQ - _qTable[state][action]);
    }
    
    // ========================================
    // [고급] Q-Table 관리 메서드
    // ========================================
    
    /// <summary>
    /// [고급] 최고 성과 Q-Table 저장
    /// 유전 알고리즘처럼 좋은 정책을 보존
    /// </summary>
    public void SaveBestQTable(float score)
    {
        if (score > _bestScore)
        {
            _bestScore = score;
            _bestQTable = new Dictionary<(int, int), float[]>();
            foreach (var kvp in _qTable)
                _bestQTable[kvp.Key] = (float[])kvp.Value.Clone();
        }
    }
    
    /// <summary>
    /// [고급] 최고 성과 Q-Table 복원
    /// 학습이 잘못된 방향으로 가면 이전 최고 정책으로 되돌림
    /// </summary>
    public void RestoreBestQTable()
    {
        if (_bestQTable == null) return;
        
        _qTable = new Dictionary<(int, int), float[]>();
        foreach (var kvp in _bestQTable)
            _qTable[kvp.Key] = (float[])kvp.Value.Clone();
    }
    
    // ========================================
    // Epsilon 관리
    // ========================================
    
    public void SetEpsilon(float value) => _epsilon = Mathf.Clamp01(value);
    public float GetEpsilon() => _epsilon;
}
