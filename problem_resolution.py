import gurobipy as gp
from gurobipy import GRB

def preprocess_data(data):
    """
    Preprocessing function that classifies keys according to their contribution.
    
    Args:
        data: dictionary with keys and their contributions
        
    Returns:
        tuple: (pros, cons, neutral) where:
            - pros: list of keys with positive contribution
            - cons: list of keys with negative contribution
            - neutral: list of keys with zero contribution
    """
    pros = []
    cons = []
    neutral = []
    
    for key, value in data.items():
        if value > 0:
            pros.append(key)
        elif value < 0:
            cons.append(key)
        else:  # value == 0
            neutral.append(key)
    
    return pros, cons, neutral

def one_to_one_tradeoffs(data, pros, cons):
    """
    Optimization with Gurobi to find one-to-one tradeoffs between pros and cons.
    
    Args:
        data: dictionary with keys and their contributions
        pros: list of keys with positive contribution
        cons: list of keys with negative contribution
        
    Returns:
        list: list of tuples (pro_key, con_key) for selected pairs,
              or 'certificate of non-existence' if the problem is infeasible
    """
    # Create the model
    model = gp.Model("one_to_one_tradeoffs")
    model.setParam('OutputFlag', 0)  # Disable Gurobi logs
    
    # Binary variables a_ij
    a = {}
    for i in range(len(pros)):
        for j in range(len(cons)):
            a[i, j] = model.addVar(vtype=GRB.BINARY, name=f"a_{i}_{j}")
    
    # Constraint C1: for fixed i, sum over j of a_ij <= 1
    for i in range(len(pros)):
        model.addConstr(gp.quicksum(a[i, j] for j in range(len(cons))) <= 1, name=f"C1_{i}")
    
    # Constraint C2: for fixed j, sum over i of a_ij = 1
    for j in range(len(cons)):
        model.addConstr(gp.quicksum(a[i, j] for i in range(len(pros))) == 1, name=f"C2_{j}")
    
    # Constraint C3: data[pros[i]] + a_ij * data[cons[j]] >= 0
    # Note: when a_ij = 1, this becomes data[pros[i]] + data[cons[j]] >= 0
    for i in range(len(pros)):
        for j in range(len(cons)):
            model.addConstr(
                data[pros[i]] + a[i, j] * data[cons[j]] >= 0,
                name=f"C3_{i}_{j}"
            )
    """
    # Objective function: max sum over i and j of a_ij * (data[pros[i]] + data[cons[j]])
    model.setObjective(
        gp.quicksum(a[i, j] * (data[pros[i]] + data[cons[j]]) 
                   for i in range(len(pros)) 
                   for j in range(len(cons))),
        GRB.MAXIMIZE
    )
    """
    # Objective function: max sum over i and j of a_ij * (data[pros[i]] + data[cons[j]])
    model.setObjective(
        gp.quicksum(a[i, j] 
                   for i in range(len(pros)) 
                   for j in range(len(cons))),
        GRB.MAXIMIZE
    )

    # Optimize
    model.optimize()
    
    # Check solution status
    if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
        return 'certificate of non-existence'
    
    # Extract pairs where a_ij = 1
    couples = []
    for i in range(len(pros)):
        for j in range(len(cons)):
            if a[i, j].x > 0.5:  # Binary variable, check if close to 1
                couples.append((pros[i], cons[j]))
    
    return couples

def one_to_m_tradeoffs(data, pros, cons):
    """
    Optimization with Gurobi to find one-to-many tradeoffs between pros and cons.
    Similar to one_to_one_tradeoffs but without constraint C1 and with modified C3.
    
    Args:
        data: dictionary with keys and their contributions
        pros: list of keys with positive contribution
        cons: list of keys with negative contribution
        
    Returns:
        list: list of tuples (pro_key, tuple of con_keys) for selected pairs,
              or 'certificate of non-existence' if the problem is infeasible
    """
    # Create the model
    model = gp.Model("one_to_m_tradeoffs")
    model.setParam('OutputFlag', 0)  # Disable Gurobi logs
    
    # Binary variables a_ij
    a = {}
    for i in range(len(pros)):
        for j in range(len(cons)):
            a[i, j] = model.addVar(vtype=GRB.BINARY, name=f"a_{i}_{j}")
    
    # Constraint C1 is removed
    
    # Constraint C2: for fixed j, sum over i of a_ij = 1
    for j in range(len(cons)):
        model.addConstr(gp.quicksum(a[i, j] for i in range(len(pros))) == 1, name=f"C2_{j}")
    
    # Constraint C3: data[pros[i]] + sum over j of a_ij * data[cons[j]] >= 0
    for i in range(len(pros)):
        model.addConstr(
            data[pros[i]] + gp.quicksum(a[i, j] * data[cons[j]] for j in range(len(cons))) >= 0,
            name=f"C3_{i}"
        )
    
    # Objective function: max sum over i and j of a_ij * (data[pros[i]] + data[cons[j]])
    model.setObjective(
        gp.quicksum(a[i, j] * (data[pros[i]] + data[cons[j]]) 
                   for i in range(len(pros)) 
                   for j in range(len(cons))),
        GRB.MAXIMIZE
    )
    
    # Optimize
    model.optimize()
    
    # Check solution status
    if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
        return 'certificate of non-existence'
    
    # Extract pairs grouped by pro_key: [('A', ('B','C','E')), ('G', ('B',))]
    result = []
    for i in range(len(pros)):
        con_keys = []
        for j in range(len(cons)):
            if a[i, j].x > 0.5:  # Binary variable, check if close to 1
                con_keys.append(cons[j])
        if con_keys:  # Only add if there are cons associated with this pro
            # Always return a tuple, even for a single element
            result.append((pros[i], tuple(con_keys)))
    
    return result


if __name__ == "__main__":
    #data = {'A': 32,'B':0,'C':-28,'D':36,'E':48,'F':-35,'G':-42}
    data = {'A': 22,'B':-2,'C':-5,'D':0,'E':-15,'F':-2,'G':2}

    pros, cons, neutral = preprocess_data(data)

    print(f"Pros (positive contribution): {pros}")
    print(f"Cons (negative contribution): {cons}")
    print(f"Neutral (zero contribution): {neutral}")

    # Question 1
    result_q1 = one_to_one_tradeoffs(data, pros, cons)
    print(f"Result one_to_one_tradeoffs: {result_q1}")

    # Question 2
    result_q2 = one_to_m_tradeoffs(data, pros, cons)
    print(f"Result one_to_m_tradeoffs: {result_q2}")



    