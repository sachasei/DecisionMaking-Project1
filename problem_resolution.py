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

def m_to_one_tradeoffs(data, pros, cons):
    """
    Optimization with Gurobi to find many-to-one tradeoffs between pros and cons.
    
    Args:
        data: dictionary with keys and their contributions
        pros: list of keys with positive contribution
        cons: list of keys with negative contribution
        
    Returns:
        list: list of tuples (tuple of pro_keys, con_key) for selected pairs,
              or 'certificate of non-existence' if the problem is infeasible
    """
    # Create the model
    model = gp.Model("m_to_1_tradeoffs")
    model.setParam('OutputFlag', 0)  # Disable Gurobi logs
    
    # Binary variables a_ij
    a = {}
    for i in range(len(pros)):
        for j in range(len(cons)):
            a[i, j] = model.addVar(vtype=GRB.BINARY, name=f"a_{i}_{j}")
    
    # Constraint C1: for fixed i, sum over j of a_ij <= 1
    for i in range(len(pros)):
        model.addConstr(gp.quicksum(a[i, j] for j in range(len(cons))) <= 1, name=f"C1_{i}")
    
    # Constraint C2: for fixed j, sum over i of a_ij * data[pros[i]] + data[cons[j]] >= 0
    for j in range(len(cons)):
        model.addConstr(
            gp.quicksum(a[i, j] * data[pros[i]] for i in range(len(pros))) + data[cons[j]] >= 0,
            name=f"C2_{j}"
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
    
    # Extract pairs grouped by con_key: [(('A','B','C'), 'D'), (('E',), 'F')]
    result = []
    for j in range(len(cons)):
        pro_keys = []
        for i in range(len(pros)):
            if a[i, j].x > 0.5:  # Binary variable, check if close to 1
                pro_keys.append(pros[i])
        if pro_keys:  # Only add if there are pros associated with this con
            # Always return a tuple, even for a single element
            result.append((tuple(pro_keys), cons[j]))
    
    return result

def m_to_one_or_one_to_m_tradeoffs(data, pros, cons):
    # Initialisation du modèle
    model = gp.Model("Explanation_L3_Robust")
    model.setParam('OutputFlag', 0)  # Désactive les logs

    # Ensembles d'indices et constante Big-M pour les scores élevés
    I, J = pros, cons
    N_star = I + J
    # M_val doit être supérieur à la somme absolue des contributions 
    M_val = sum(abs(v) for v in data.values()) + 100 

    # Variables de décision (Appendice A) [cite: 705-709]
    u = model.addVars(I, J, vtype=GRB.BINARY, name="u")  # Pro i couvre Con j (1-vs-m)
    v = model.addVars(I, J, vtype=GRB.BINARY, name="v")  # Con j couvert par Pros i (m-vs-1)
    t = model.addVars(N_star, vtype=GRB.BINARY, name="t") # Pivot de l'argument
    e = model.addVar(vtype=GRB.BINARY, name="e")         # Indicateur d'existence

    # 1. Contraintes de structure (Partitionnement) [cite: 711-714]
    for i in I:
        for j in J:
            model.addConstr(u[i, j] <= t[i]) # u_ij n'existe que si i est pivot
            model.addConstr(v[i, j] <= t[j]) # v_ij n'existe que si j est pivot

    for i in I:
        # Un pro est soit pivot d'un 1-vs-m, soit membre d'un m-vs-1 [cite: 713]
        model.addConstr(gp.quicksum(v[i, j] for j in J) + t[i] <= 1)

    for j in J:
        # Chaque con est soit pivot d'un m-vs-1, soit couvert par un 1-vs-m [cite: 714]
        model.addConstr(gp.quicksum(u[i, j] for i in I) + t[j] == 1)

    # 2. Contraintes d'alignement avec Big-M (Version corrigée de A.7/A.8) [cite: 758-759]
    # Un argument n'est validé que si sa force totale est >= 0
    for i in I:
        # Type 1-vs-m : Pro i + somme des Cons j associés
        strength_1m = data[i] + gp.quicksum(data[j] * u[i, j] for j in J)
        model.addConstr(strength_1m >= -M_val * (1 - t[i]) - M_val * (1 - e))

    for j in J:
        # Type m-vs-1 : Con j + somme des Pros i associés
        strength_m1 = data[j] + gp.quicksum(data[i] * v[i, j] for i in I)
        model.addConstr(strength_m1 >= -M_val * (1 - t[j]) - M_val * (1 - e))

    # 3. Objectif : Maximiser l'existence, puis minimiser le nombre de pivots (groupes) [cite: 760]
    # Minimiser t_k permet d'avoir des explications plus courtes et concises [cite: 382, 500]
    model.setObjective(1000 * e - gp.quicksum(t[k] for k in N_star), GRB.MAXIMIZE)
    model.optimize()
    
    if e.X < 0.5 or model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
        return 'certificate of non-existence'  # Certificat de non-existence [cite: 34, 709]

    # Extraction et formatage selon votre exemple
    results = []
    for i in I:
        if t[i].X > 0.5:
            covered_cons = tuple(sorted([j for j in J if u[i, j].X > 0.5]))
            # Format: ((Pro,), (Cons,)) ou ((Pro,), Con)
            res_cons = covered_cons[0] if len(covered_cons) == 1 else covered_cons
            results.append(((i,), res_cons))
            
    for j in J:
        if t[j].X > 0.5:
            covering_pros = tuple(sorted([i for i in I if v[i, j].X > 0.5]))
            results.append((covering_pros, j))

    return results


if __name__ == "__main__":
    #data = {'A': 32,'B':0,'C':-28,'D':36,'E':48,'F':-35,'G':-42}
    #data = {'A': 22,'B':-2,'C':-5,'D':0,'E':-15,'F':-2,'G':2}
    #data= {'A':8,'B':14,'C':21,'D':-42,'E':72,'F':-65,'G':0} # u > v
    #data = {'A':49,'B':-56,'C':7,'D':-48,'E':-6,'F':20,'G':96} # y > z
    data= {'A':0,'B':126,'C':-70,'D':-60,'E':54,'F':-40,'G':36} # z > t

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

    # Question 3
    result_q3 = m_to_one_tradeoffs(data, pros, cons)
    print(f"Result m_to_one_tradeoffs: {result_q3}")

    # Question 4
    result_q4 = m_to_one_or_one_to_m_tradeoffs(data, pros, cons)
    print(f"Result m_to_one_or_one_to_m_tradeoffs: {result_q4}")



    