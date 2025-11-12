import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AmericanOptionPINN(nn.Module):
    """
    PINN para opciones americanas con ejercicio temprano
    """
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super(AmericanOptionPINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, S, t):
        x = torch.cat([S, t], dim=1)
        
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        
        # IMPORTANTE: Usamos softplus para garantizar V ≥ 0
        x = torch.nn.functional.softplus(self.layers[-1](x))
        return x


class AmericanOptionTrainer:
    """
    Entrenador para opciones americanas con condición de ejercicio temprano
    """
    def __init__(self, model, S_range, t_range, K, r, sigma, option_type='put'):
        self.model = model.to(device)
        self.K = K
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.S_range = S_range
        self.t_range = t_range
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.losses = []
    
    def payoff(self, S):
        """Función de payoff intrínseco"""
        if self.option_type == 'call':
            return torch.maximum(S - self.K, torch.zeros_like(S))
        else:  # put (más común para americanas)
            return torch.maximum(self.K - S, torch.zeros_like(S))
    
    def black_scholes_pde(self, S, t):
        """
        Calcula el residuo de la PDE
        """
        S.requires_grad_(True)
        t.requires_grad_(True)
        
        V = self.model(S, t)
        
        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                   create_graph=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                   create_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                    create_graph=True)[0]
        
        # Ecuación de Black-Scholes
        pde_residual = (V_t + 0.5 * self.sigma**2 * S**2 * V_SS + 
                       self.r * S * V_S - self.r * V)
        
        return pde_residual, V
    
    def sample_points(self, n_interior, n_boundary, n_terminal):
        """
        Genera puntos de entrenamiento
        """
        S_min, S_max = self.S_range
        t_min, T = self.t_range
        
        # Puntos interiores
        S_int = torch.rand(n_interior, 1) * (S_max - S_min) + S_min
        t_int = torch.rand(n_interior, 1) * (T - t_min) + t_min
        
        # Puntos terminales (t = T)
        S_term = torch.rand(n_terminal, 1) * (S_max - S_min) + S_min
        t_term = torch.ones(n_terminal, 1) * T
        
        # Fronteras
        S_bound_low = torch.zeros(n_boundary, 1)
        t_bound_low = torch.rand(n_boundary, 1) * (T - t_min) + t_min
        
        S_bound_high = torch.ones(n_boundary, 1) * S_max
        t_bound_high = torch.rand(n_boundary, 1) * (T - t_min) + t_min
        
        return (S_int.to(device), t_int.to(device), 
                S_term.to(device), t_term.to(device),
                S_bound_low.to(device), t_bound_low.to(device),
                S_bound_high.to(device), t_bound_high.to(device))
    
    def loss_function(self, S_int, t_int, S_term, t_term, 
                     S_bound_low, t_bound_low, S_bound_high, t_bound_high):
        """
        Loss function para opciones americanas con condición de ejercicio temprano
        """
        
        # 1. LOSS DE PDE (interior)
        pde_residual, V_int = self.black_scholes_pde(S_int, t_int)
        payoff_int = self.payoff(S_int)
        
        # CLAVE: Para opciones americanas, la PDE solo se cumple donde V > payoff
        # Donde V = payoff, estamos en la frontera de ejercicio
        # Usamos max(pde_residual, V - payoff) ≈ 0
        
        # Opción 1: Penalizar ambos (PDE y violación de ejercicio temprano)
        loss_pde = torch.mean(pde_residual**2)
        loss_early_exercise = torch.mean(torch.relu(payoff_int - V_int)**2)
        
        # 2. LOSS DE CONDICIÓN TERMINAL
        V_term = self.model(S_term, t_term)
        payoff_term = self.payoff(S_term)
        loss_terminal = torch.mean((V_term - payoff_term)**2)
        
        # 3. LOSS DE FRONTERA INFERIOR
        V_bound_low = self.model(S_bound_low, t_bound_low)
        if self.option_type == 'put':
            # Para put: V(0, t) = K * exp(-r*(T-t))
            V_expected_low = self.K * torch.exp(-self.r * (self.t_range[1] - t_bound_low))
        else:
            V_expected_low = torch.zeros_like(V_bound_low)
        loss_bound_low = torch.mean((V_bound_low - V_expected_low)**2)
        
        # 4. LOSS DE FRONTERA SUPERIOR
        V_bound_high = self.model(S_bound_high, t_bound_high)
        if self.option_type == 'call':
            V_expected_high = S_bound_high - self.K * torch.exp(-self.r * (self.t_range[1] - t_bound_high))
        else:
            # Para put americana: V → 0 cuando S → ∞
            V_expected_high = torch.zeros_like(V_bound_high)
        loss_bound_high = torch.mean((V_bound_high - V_expected_high)**2)
        
        # LOSS TOTAL con pesos ajustados para opciones americanas
        total_loss = (1.0 * loss_pde + 
                     20.0 * loss_early_exercise +  # MUY IMPORTANTE
                     10.0 * loss_terminal + 
                     1.0 * loss_bound_low + 
                     1.0 * loss_bound_high)
        
        return total_loss, loss_pde, loss_early_exercise, loss_terminal, loss_bound_low, loss_bound_high
    
    def train(self, epochs, n_interior=2000, n_boundary=100, n_terminal=100):
        """
        Loop de entrenamiento
        """
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            points = self.sample_points(n_interior, n_boundary, n_terminal)
            
            total_loss, loss_pde, loss_ee, loss_term, loss_low, loss_high = self.loss_function(*points)
            
            total_loss.backward()
            self.optimizer.step()
            
            self.losses.append(total_loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  PDE Loss: {loss_pde.item():.6f}")
                print(f"  Early Exercise Loss: {loss_ee.item():.6f}")
                print(f"  Terminal Loss: {loss_term.item():.6f}")
                print(f"  Boundary Losses: {loss_low.item():.6f}, {loss_high.item():.6f}")
                print("-" * 50)
    
    def predict(self, S, t):
        """
        Predice el precio de la opción
        """
        self.model.eval()
        with torch.no_grad():
            S_tensor = torch.FloatTensor(S).reshape(-1, 1).to(device)
            t_tensor = torch.FloatTensor(t).reshape(-1, 1).to(device)
            V = self.model(S_tensor, t_tensor)
        return V.cpu().numpy()
    
    def find_early_exercise_boundary(self, t_eval, n_points=200):
        """
        Encuentra la frontera de ejercicio temprano
        """
        S_test = np.linspace(self.S_range[0], self.S_range[1], n_points)
        t_test = np.ones_like(S_test) * t_eval
        
        V_pred = self.predict(S_test, t_test).flatten()
        payoff = self.payoff(torch.FloatTensor(S_test).reshape(-1, 1)).cpu().numpy().flatten()
        
        # Frontera de ejercicio: donde V ≈ payoff
        threshold = 0.01  # Tolerancia
        exercise_region = np.abs(V_pred - payoff) < threshold
        
        return S_test, V_pred, payoff, exercise_region


# ==================== EJEMPLO DE USO ====================
if __name__ == "__main__":
    # Parámetros típicos para una PUT americana
    K = 100.0      # Strike
    r = 0.05       # Risk-free rate
    sigma = 0.3    # Volatilidad (30% - más alta para ver ejercicio temprano)
    T = 1.0        # Maturity
    
    S_range = [40.0, 160.0]
    t_range = [0.0, T]
    
    # Crear modelo
    model = AmericanOptionPINN(layers=[2, 64, 64, 64, 1])
    
    # Crear trainer para PUT americana
    trainer = AmericanOptionTrainer(
        model=model,
        S_range=S_range,
        t_range=t_range,
        K=K,
        r=r,
        sigma=sigma,
        option_type='put'  # PUT es más interesante para americanas
    )
    
    # Entrenar
    print("Entrenando PINN para Opción Americana...")
    trainer.train(epochs=3000, n_interior=3000, n_boundary=200, n_terminal=200)
    
    # Visualización: Precio de la opción en diferentes tiempos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = [0.0, 0.3, 0.6, 0.9]
    
    for idx, t_eval in enumerate(times):
        ax = axes[idx // 2, idx % 2]
        
        S_test, V_pred, payoff, exercise_region = trainer.find_early_exercise_boundary(t_eval)
        
        # Graficar precio de opción vs payoff
        ax.plot(S_test, V_pred, label=f'Precio Opción Americana (t={t_eval:.1f})', linewidth=2, color='blue')
        ax.plot(S_test, payoff, '--', label='Payoff Intrínseco', linewidth=2, color='red')
        
        # Sombrear región de ejercicio temprano
        ax.fill_between(S_test, 0, max(V_pred)*1.2, where=exercise_region, 
                        alpha=0.3, color='green', label='Región de Ejercicio')
        
        ax.set_xlabel('Precio del Activo (S)', fontsize=10)
        ax.set_ylabel('Valor de la Opción (V)', fontsize=10)
        ax.set_title(f'Tiempo hasta vencimiento: {T - t_eval:.1f} años', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(V_pred)*1.1])
    
    plt.suptitle(f'Opción PUT Americana: K=${K}, r={r}, σ={sigma}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('american_option_pinn.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Curva de aprendizaje
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.losses)
    plt.xlabel('Época')
    plt.ylabel('Loss Total')
    plt.title('Curva de Aprendizaje')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Comparación en t=0.5
    plt.subplot(1, 2, 2)
    S_test = np.linspace(40, 160, 100)
    t_test = np.ones_like(S_test) * 0.5
    V_american = trainer.predict(S_test, t_test).flatten()
    payoff = np.maximum(K - S_test, 0)
    
    plt.plot(S_test, V_american, label='Americana PINN', linewidth=2)
    plt.plot(S_test, payoff, '--', label='Payoff', linewidth=2)
    plt.fill_between(S_test, payoff, V_american, alpha=0.3, label='Time Value')
    plt.xlabel('S')
    plt.ylabel('V')
    plt.title('Valor Temporal vs Intrínseco (t=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('american_option_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("✅ Entrenamiento completado!")
    print("="*60)
    
    # Mostrar frontera de ejercicio en t=0.5
    S_test, V_pred, payoff, exercise_region = trainer.find_early_exercise_boundary(0.5)
    exercise_boundary = S_test[exercise_region]
    if len(exercise_boundary) > 0:
        print(f"\n📊 En t=0.5 años antes del vencimiento:")
        print(f"   Frontera de ejercicio: S ≈ ${exercise_boundary.min():.2f} - ${exercise_boundary.max():.2f}")
        print(f"   Si S < ${exercise_boundary.min():.2f}, conviene ejercer la put")
        print(f"   Si S > ${exercise_boundary.max():.2f}, conviene esperar")
    
    print("\n💡 Nota: Las opciones PUT americanas se ejercen temprano cuando")
    print("   el activo cae mucho, porque el valor intrínseco (K-S) es mayor")
    print("   que el valor de esperar (debido al factor de descuento).")