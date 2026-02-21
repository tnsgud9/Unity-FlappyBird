using UnityEngine;

/// <summary>
/// 플레이어 에이전트 - Q-Learning으로 학습하여 Flappy Bird 플레이
/// 
/// [강화학습 사이클]
/// 1. 관측 (Observation): 현재 상태 파악
/// 2. 행동 (Action): Q-Learning으로 행동 결정
/// 3. 보상 (Reward): 행동의 결과 받기
/// 4. 학습 (Learning): Q-Value 갱신
/// 
/// [상태 정의]
/// - distY: 파이프 갭 중심과 새의 Y 거리 (0~9)
/// - pipeDistX: 가장 가까운 파이프까지의 X 거리 (0~5)
/// 
/// [행동 정의]
/// - 0: 대기 (가만히 있기)
/// - 1: 점프 (위로 이동)
/// 
/// [보상 구조]
/// - 생존: +surviveReward (매 프레임)
/// - 파이프 갭 근접: +gapProximityReward × (1 - 거리/5)
/// - 파이프 통과: +pipePassReward
/// - 충돌: +collisionPenalty
/// </summary>
public class PlayerAgent : MonoBehaviour
{
    // ========================================
    // Inspector 설정
    // ========================================
    
    [Header("물리 설정")]
    public float jumpForce = 5f;              // 점프 힘
    public float jumpCooldownTime = 0.2f;     // 점프 쿨다운 (연속 점프 방지)
    
    [Header("Q-Learning 파라미터")]
    public float learningRate = 0.3f;         // 학습률 (α)
    public float discountFactor = 0.9f;       // 할인율 (γ)
    public float initialEpsilon = 1f;         // 초기 탐색률 (ε)
    
    [Header("보상 설정")]
    public float surviveReward = 0.01f;       // 생존 보상
    public float pipePassReward = 1f;         // 파이프 통과 보상
    public float collisionPenalty = -1f;      // 충돌 페널티
    public float gapProximityReward = 0.3f;   // 갭 근접 보상 계수
    
    [Header("행동 제약 (안전장치)")]
    public float maxYPosition = 8f;           // 이 이상이면 점프 금지
    public float minYPosition = 2f;           // 이 이하이면 점프 강제
    public float maxUpVelocity = 1f;          // 상승 중이면 점프 금지
    
    // ========================================
    // 내부 변수
    // ========================================
    private Rigidbody _rigidbody;
    private AgentManager _agentManager;
    private QLearning _qLearning;
    private Vector3 _initialPosition;
    private (int, int) _currentState;
    private bool _isDead;
    private float _currentReward;
    private int _currentScore;
    private float _jumpCooldown;
    
    public int Score => _currentScore;
    
    // ========================================
    // 초기화
    // ========================================
    void Start()
    {
        _rigidbody = GetComponent<Rigidbody>();
        _agentManager = FindFirstObjectByType<AgentManager>();
        
        // Q-Learning 초기화
        _qLearning = new QLearning(learningRate, discountFactor, initialEpsilon);
        
        _initialPosition = transform.position;
        _currentState = GetState();
        _isDead = false;
        _jumpCooldown = 0f;
    }
    
    // ========================================
    // 메인 루프: 강화학습 사이클
    // ========================================
    void FixedUpdate()
    {
        if (_isDead) return;
        
        _jumpCooldown -= Time.fixedDeltaTime;
        
        // [1] 관측: 현재 상태 파악
        (int, int) nextState = GetState();
        
        // [2] 행동: Q-Learning으로 행동 결정
        int action = _qLearning.GetAction(nextState);
        
        // 행동 제약 적용 (물리적 안전장치)
        action = ApplyActionConstraints(action);
        
        // 행동 실행
        if (action == 1 && _jumpCooldown <= 0f)
        {
            Jump();
            _jumpCooldown = jumpCooldownTime;
        }
        
        // [3] 보상: 행동의 결과 계산
        CalculateReward();
        _currentReward += surviveReward;
        
        // [4] 학습: Q-Value 갱신
        _qLearning.Update(_currentState, action, _currentReward, nextState);
        
        // 다음 프레임 준비
        _currentState = nextState;
        _currentReward = 0f;
    }
    
    // ========================================
    // [1] 상태 관측
    // ========================================
    
    /// <summary>
    /// 현재 상태 반환
    /// 상태 = (distY, pipeDistX)
    /// </summary>
    (int, int) GetState()
    {
        GameObject closestPipe = FindClosestPipe();
        int distY = 5;      // 기본값: 갭 중심
        int pipeDistX = 5;  // 기본값: 파이프 없음
        
        if (closestPipe != null)
        {
            // 파이프 갭 중심과의 Y 거리
            float gapCenterY = GetGapCenterY(closestPipe);
            distY = Mathf.Clamp(Mathf.FloorToInt((transform.position.y - gapCenterY) / 2f), -5, 4) + 5;
            
            // 파이프까지의 X 거리
            float distX = closestPipe.transform.position.x - transform.position.x;
            pipeDistX = Mathf.Clamp(Mathf.FloorToInt(distX / 2f), 0, 5);
        }
        
        return (distY, pipeDistX);
    }
    
    /// <summary>
    /// 가장 가까운 파이프 찾기 (앞에 있는 것만)
    /// </summary>
    GameObject FindClosestPipe()
    {
        PipeMovement[] pipes = FindObjectsByType<PipeMovement>(FindObjectsSortMode.None);
        GameObject closest = null;
        float minDist = Mathf.Infinity;
        
        foreach (var pipe in pipes)
        {
            float dist = pipe.transform.position.x - transform.position.x;
            
            // 앞에 있는 파이프만 고려
            if (dist > 0 && dist < minDist)
            {
                minDist = dist;
                closest = pipe.gameObject;
            }
        }
        
        return closest;
    }
    
    /// <summary>
    /// 파이프의 갭(통과 구간) 중심 Y 좌표 계산
    /// </summary>
    float GetGapCenterY(GameObject pipe)
    {
        if (pipe.transform.childCount >= 2)
        {
            // 자식 0: 위 파이프, 자식 1: 아래 파이프
            float topY = pipe.transform.GetChild(0).position.y;
            float bottomY = pipe.transform.GetChild(1).position.y;
            return (topY + bottomY) / 2f;
        }
        return pipe.transform.position.y;
    }
    
    // ========================================
    // [2] 행동 실행
    // ========================================
    
    /// <summary>
    /// 점프 실행
    /// </summary>
    void Jump()
    {
        _rigidbody.linearVelocity = Vector3.up * jumpForce;
    }
    
    /// <summary>
    /// 행동 제약 적용 (물리적 안전장치)
    /// </summary>
    int ApplyActionConstraints(int action)
    {
        float y = transform.position.y;
        float velocityY = _rigidbody.linearVelocity.y;
        
        // 너무 높거나 상승 중이면 점프 금지
        if (y > maxYPosition || velocityY > maxUpVelocity)
            return 0;
        
        // 너무 낮고 하강 중이면 점프 강제
        if (y < minYPosition && velocityY < 0f)
            return 1;
        
        return action;
    }
    
    // ========================================
    // [3] 보상 계산
    // ========================================
    
    /// <summary>
    /// 보상 계산: 파이프 갭에 가까울수록 보상
    /// </summary>
    void CalculateReward()
    {
        GameObject closestPipe = FindClosestPipe();
        
        if (closestPipe != null)
        {
            float gapCenterY = GetGapCenterY(closestPipe);
            float distToGap = Mathf.Abs(transform.position.y - gapCenterY);
            
            // 갭에 가까울수록 높은 보상 (0 ~ gapProximityReward)
            float gapReward = Mathf.Max(0, 1f - distToGap / 5f);
            _currentReward += gapReward * gapProximityReward;
        }
    }
    
    /// <summary>
    /// 충돌 시: 큰 페널티 + 에피소드 종료
    /// </summary>
    void OnCollisionEnter(Collision collision)
    {
        _isDead = true;
        _currentReward = collisionPenalty;
        
        // 마지막 학습
        _qLearning.Update(_currentState, 0, _currentReward, _currentState);
        
        // 에피소드 종료
        _agentManager.EndEpisode();
    }
    
    /// <summary>
    /// 파이프 통과 시: 큰 보상
    /// </summary>
    void OnTriggerExit(Collider other)
    {
        _currentReward += pipePassReward;
        _currentScore++;
    }
    
    // ========================================
    // [4] 에피소드 관리
    // ========================================
    
    /// <summary>
    /// 에피소드 리셋
    /// </summary>
    public void Reset()
    {
        _rigidbody.linearVelocity = Vector3.zero;
        transform.position = _initialPosition;
        _isDead = false;
        _currentReward = 0f;
        _currentScore = 0;
        _jumpCooldown = 0f;
        _currentState = GetState();
    }
    
    // ========================================
    // Q-Learning 연동
    // ========================================
    
    public void SetEpsilon(float value) => _qLearning.SetEpsilon(value);
    public float GetEpsilon() => _qLearning.GetEpsilon();
    
    // ========================================
    // [고급] Best Q-Table 관리
    // ========================================
    
    public void SaveBestQTable(float score) => _qLearning.SaveBestQTable(score);
    public void RestoreBestQTable() => _qLearning.RestoreBestQTable();
}
