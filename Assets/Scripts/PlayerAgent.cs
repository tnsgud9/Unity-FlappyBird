using UnityEngine;

/// <summary>
/// ============================================================================
/// 플레이어 에이전트 - Q-Learning으로 학습하여 Flappy Bird 플레이
/// ============================================================================
/// 
/// [강화학습 사이클]
/// 1. 관측 (Observation): 현재 상태 파악
/// 2. 행동 (Action): Q-Learning으로 행동 결정
/// 3. 보상 (Reward): 행동의 결과 받기
/// 4. 학습 (Learning): Q-Value 갱신
/// 
/// 이 4단계가 매 프레임 반복됩니다!
/// 
/// ============================================================================
/// [상태 정의]
/// ============================================================================
/// 
/// stateY (0~9): 새의 Y 위치
///   - 0 = 아래쪽, 9 = 위쪽
///   - 실제 Y 좌표를 10단계로 나눔
///   - 예: Y=3.5 → stateY = 4
/// 
/// distY (0~9): 파이프 갭 중심과의 Y 거리
///   - 5 = 갭 중심에 정확히 위치
///   - 0 = 갭보다 아래, 9 = 갭보다 위
///   - 예: 새 Y=7, 갭 중심 Y=5 → distY = 5 (중심!)
/// 
/// pipeDistX (0~5): 가장 가까운 파이프까지의 X 거리
///   - 0 = 파이프가 매우 가까움
///   - 5 = 파이프가 멀거나 없음
///   - 예: 파이프가 4칸 앞 → pipeDistX = 2
/// 
/// ============================================================================
/// [행동 정의]
/// ============================================================================
/// 
/// 0: 대기 (가만히 있기)
///   - 새가 떨어지지 않음
///   - 중력에 의해 자연스럽게 떨어짐
/// 
/// 1: 점프 (위로 이동)
///   - 새가 위로 빠르게 이동
///   - 점프력(jumpForce)에 의해 이동 속도 결정
/// 
/// ============================================================================
/// [보상 구조 - 통합 학습]
/// ============================================================================
/// 
/// 모든 보상 요소를 동시에 고려하여 학습합니다.
/// 
/// 생존 보상 (+surviveReward)
///   - 매 프레임 살아있으면 보상
///   - "오래 생존할수록" 학습됩니다
/// 
/// 갭 근접 보상 (+gapProximityReward × 거리)
///   - 파이프 갭에 가까울수록 보상
///   - 거리 = 0 → 최대 보상
///   - 거리 = 5 → 보상 없음
/// 
/// 갭 중심 유지 보상 (+gapCenterReward)
///   - 갭 중심에 정확히 위치하면 추가 보상
///   - distY = 5일 때 최대 보상
/// 
/// 파이프 통과 보상 (+pipePassReward)
///   - 파이프를 통과하면 큰 보상
///   - 성공의 신호!
/// 
/// 충돌 페널티 (+collisionPenalty)
///   - 장애물에 부딪히면 큰 페널티
///   - "이러면 안 돼" 학습
/// </summary>
public class PlayerAgent : MonoBehaviour
{
    // ========================================
    // Inspector 설정 (유니티 에디터에서 조절 가능)
    // ========================================
    
    [Header("물리 설정")]
    [Tooltip("점프 힘: 높을수록 더 높게 점프")]
    public float jumpForce = 5f;
    
    [Tooltip("점프 쿨다운: 연속 점프 방지 (초당 최대 5회)")]
    public float jumpCooldownTime = 0.2f;
    
    [Header("Q-Learning 파라미터")]
    [Tooltip("학습률: 높을수록 빠른 학습")]
    public float learningRate = 0.3f;
    
    [Tooltip("할인율: 높을수록 미래 중시")]
    public float discountFactor = 0.9f;
    
    [Tooltip("초기 탐색률: 처음에는 100% 탐색")]
    public float initialEpsilon = 1f;
    
    [Header("보상 설정")]
    [Tooltip("생존 보상: 매 프레임")]
    public float surviveReward = 0.01f;
    
    [Tooltip("갭 근접 보상: 갭에 가까울수록")]
    public float gapProximityReward = 0.3f;
    
    [Tooltip("갭 중심 보상: 중심에 있을수록")]
    public float gapCenterReward = 0.1f;
    
    [Tooltip("파이프 통과 보상")]
    public float pipePassReward = 1f;
    
    [Tooltip("충돌 페널티")]
    public float collisionPenalty = -1f;
    
    [Header("행동 제약 (안전장치)")]
    [Tooltip("이 높이 이상이면 점프 금지")]
    public float maxYPosition = 8f;
    
    [Tooltip("이 높이 이하이면 점프 강제")]
    public float minYPosition = 2f;
    
    [Tooltip("상승 속도가 이 이상이면 점프 금지")]
    public float maxUpVelocity = 1f;
    
    // ========================================
    // 내부 변수
    // ========================================
    private Rigidbody _rigidbody;          // 물리 컴포넌트
    private AgentManager _agentManager;      // 학습 관리자
    private QLearning _qLearning;          // Q-Learning 알고리즘
    private Vector3 _initialPosition;       // 초기 위치 (리셋용)
    private (int, int, int) _currentState; // 현재 상태
    private bool _isDead;               // 사망 여부
    private float _currentReward;         // 현재 프레임 보상
    private int _currentScore;            // 현재 점수
    private float _jumpCooldown;          // 점프 쿨다운
    
    public int Score => _currentScore;
    
    // ========================================
    // 초기화
    // ========================================
    void Start()
    {
        // 컴포넌트 참조
        _rigidbody = GetComponent<Rigidbody>();
        
        // AgentManager 찾기
        _agentManager = FindFirstObjectByType<AgentManager>();
        
        // Q-Learning 초기화
        _qLearning = new QLearning(learningRate, discountFactor, initialEpsilon);
        
        // 초기 위치 저장 (리셋용)
        _initialPosition = transform.position;
        
        // 초기 상태 설정
        _currentState = GetState();
        
        // 상태 초기화
        _isDead = false;
        _jumpCooldown = 0f;
    }
    
    // ========================================
    // 메인 루프: 강화학습 사이클
    // ========================================
    void FixedUpdate()
    {
        // 사망 상태면 실행 안 함
        if (_isDead) return;
        
        // 점프 쿨다운 감소
        _jumpCooldown -= Time.fixedDeltaTime;
        
        // ========================================
        // [1] 관측: 현재 상태 파악
        // ========================================
        (int, int, int) nextState = GetState();
        
        // ========================================
        // [2] 행동: Q-Learning으로 행동 결정
        // ========================================
        int action = _qLearning.GetAction(nextState);
        
        // 행동 제약 적용 (안전장치)
        action = ApplyActionConstraints(action);
        
        // ========================================
        // 행동 실행
        // ========================================
        if (action == 1 && _jumpCooldown <= 0f)
        {
            Jump();
            _jumpCooldown = jumpCooldownTime;
        }
        
        // ========================================
        // [3] 보상: 행동의 결과 계산
        // ========================================
        CalculateReward();
        _currentReward += surviveReward;
        
        // ========================================
        // [4] 학습: Q-Value 갱신
        // ========================================
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
    /// 상태 = (stateY, distY, pipeDistX)
    /// </summary>
    (int, int, int) GetState()
    {
        // 새의 Y 위치 계산 (0~9)
        // transform.position.y: 새의 실제 Y 좌표
        // + 5: Y가 음수일 때 0, 양수일 때 양수
        // / 2: 2단위로 나누어 0~9 사이의 값으로 변환
        int stateY = Mathf.Clamp(Mathf.FloorToInt((transform.position.y + 5f) / 2f), 0, 9);
        
        // 가장 가까운 파이프 찾기
        GameObject closestPipe = FindClosestPipe();
        
        // 기본값 설정 (파이프가 없을 때)
        int distY = 5;      // 갭 중심
        int pipeDistX = 5;  // 파이프 없음
        
        if (closestPipe != null)
        {
            // ========================================
            // 파이프 갭 중심과의 Y 거리 계산
            // ========================================
            float gapCenterY = GetGapCenterY(closestPipe);
            
            // distY 계산: 새 위치 - 갭 중심
            // -5~4 범위를 0~9로 변환
            distY = Mathf.Clamp(Mathf.FloorToInt((transform.position.y - gapCenterY) / 2f), -5, 4) + 5;
            
            // ========================================
            // 파이프까지의 X 거리 계산
            // ========================================
            float distX = closestPipe.transform.position.x - transform.position.x;
            
            // 0~5 범위로 변환 (0=매우 가까움, 5=멀거나 없음)
            pipeDistX = Mathf.Clamp(Mathf.FloorToInt(distX / 2f), 0, 5);
        }
        
        return (stateY, distY, pipeDistX);
    }
    
    /// <summary>
    /// 가장 가까운 파이프 찾기
    /// 앞에 있는 파이프만 고려 (이미 지나간 파이프는 무시)
    /// </summary>
    GameObject FindClosestPipe()
    {
        // 모든 파이프 찾기
        PipeMovement[] pipes = FindObjectsByType<PipeMovement>(FindObjectsSortMode.None);
        GameObject closest = null;
        float minDist = Mathf.Infinity;
        
        foreach (var pipe in pipes)
        {
            // X 거리 계산 (파이프 X - 새 X)
            float dist = pipe.transform.position.x - transform.position.x;
            
            // 앞에 있는 파이프만 고려 (dist > 0)
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
    /// 
    /// 파이프 구조:
    /// Pipe (부모)
    ///   ├── TopPipe (자식 0) - 위쪽 파이프
    ///   └── BottomPipe (자식 1) - 아래쪽 파이프
    /// 
    /// 갭 중심 = (TopPipe Y + BottomPipe Y) / 2
    /// </summary>
    float GetGapCenterY(GameObject pipe)
    {
        // 자식이 2개 이상이면
        if (pipe.transform.childCount >= 2)
        {
            // 위 파이프와 아래 파이프의 Y 좌표
            float topY = pipe.transform.GetChild(0).position.y;
            float bottomY = pipe.transform.GetChild(1).position.y;
            
            // 중간값 반환
            return (topY + bottomY) / 2f;
        }
        
        // 자식이 없으면 파이프 자체의 Y 좌표 사용
        return pipe.transform.position.y;
    }
    
    // ========================================
    // [2] 행동 실행
    // ========================================
    
    /// <summary>
    /// 점프 실행
    /// linearVelocity를 설정하여 즉시 위로 이동
    /// </summary>
    void Jump()
    {
        // Y축으로 점프
        _rigidbody.linearVelocity = Vector3.up * jumpForce;
    }
    
    /// <summary>
    /// 행동 제약 적용 (물리적 안전장치)
    /// 
    /// Q-Learning이 학습하기 전에 너무 멍청한 행동을 방지
    /// - 너무 높으면 점프 금지 (천장 충돌 방지)
    /// - 너무 낮으면 점프 강제 (바닥 충돌 방지)
    /// - 상승 중이면 점프 금지 (불필요한 점프 방지)
    /// </summary>
    int ApplyActionConstraints(int action)
    {
        float y = transform.position.y;
        float velocityY = _rigidbody.linearVelocity.y;
        
        // 너무 높거나 상승 중이면 점프 금지
        if (y > maxYPosition || velocityY > maxUpVelocity)
            return 0;  // 대기
        
        // 너무 낮고 하강 중이면 점프 강제
        if (y < minYPosition && velocityY < 0f)
            return 1;  // 점프
        
        return action;
    }
    
    // ========================================
    // [3] 보상 계산 - 통합 학습
    // ========================================
    
    /// <summary>
    /// 보상 계산
    /// 
    /// 세 가지 목표를 동시에 학습:
    /// 1. 생존: 오래 살아남기
    /// 2. 갭 중심 유지: 파이프 갭 중심으로 이동
    /// 3. 파이프 통과: 장애물을 피해서 통과
    /// </summary>
    void CalculateReward()
    {
        GameObject closestPipe = FindClosestPipe();
        
        if (closestPipe != null)
        {
            float gapCenterY = GetGapCenterY(closestPipe);
            float distToGap = Mathf.Abs(transform.position.y - gapCenterY);
            
            // ========================================
            // [보상 1] 갭 근접 보상
            // ========================================
            // 갭에 가까울수록 보상
            // 거리 0 → 최대 보상 (1.0)
            // 거리 5 → 보상 없음 (0.0)
            // ========================================
            float proximityReward = Mathf.Max(0, 1f - distToGap / 5f);
            _currentReward += proximityReward * gapProximityReward;
            
            // ========================================
            // [보상 2] 갭 중심 유지 보상
            // ========================================
            // distY가 5(중심)일 때 최대 보상
            // |distY - 5|가 0이면 중심 → 최대 보상
            // |distY - 5|가 5이면 끝 → 보상 없음
            // ========================================
            int distY = Mathf.Clamp(Mathf.FloorToInt((transform.position.y - gapCenterY) / 2f), -5, 4) + 5;
            float centerBonus = 1f - Mathf.Abs(distY - 5) / 5f;
            _currentReward += centerBonus * gapCenterReward;
        }
    }
    
    /// <summary>
    /// 충돌 시: 페널티 + 에피소드 종료
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
        // 물리 상태 초기화
        _rigidbody.linearVelocity = Vector3.zero;
        
        // 위치 초기화
        transform.position = _initialPosition;
        
        // 상태 초기화
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
