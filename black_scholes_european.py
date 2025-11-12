import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BlackScholesPINN(nn.Module):
    """
    Physics-Informed Neural Network para resolver la ecuación de Black-Scholes
    """
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super(BlackScholesPINN, self).__init__()
        
        # Construir la red
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Inicialización Xavier
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, S, t):
        """
        Forward pass
        Input: S (precio del activo), t (tiempo)
        Output: V (precio de la opción)
        """
        # Normalización de inputs (importante para estabilidad)
        x = torch.cat([S, t], dim=1)
        
        # Propagación a través de capas ocultas
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        
        # Capa de salida (sin activación)
        x = self.layers[-1](x)
        return x


class BlackScholesTrainer:
    """
    Entrenador para el PINN de Black-Scholes
    """
    def __init__(self, model, S_range, t_range, K, r, sigma, option_type='call'):
        self.model = model.to(device)
        self.K = K  # Strike price
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type
        self.S_range = S_range  # [S_min, S_max]
        self.t_range = t_range  # [t_min, T]
        
        # Optimizador
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Para tracking
        self.losses = []
    
    def payoff(self, S):
        """Condición terminal (payoff)"""
        if self.option_type == 'call':
            return torch.maximum(S - self.K, torch.zeros_like(S))
        else:  # put
            return torch.maximum(self.K - S, torch.zeros_like(S))
    
    def black_scholes_pde(self, S, t):
        """
        Calcula el residuo de la ecuación de Black-Scholes
        """
        S.requires_grad_(True)
        t.requires_grad_(True)
        
        V = self.model(S, t)
        
        # Derivadas usando autograd
        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                   create_graph=True)[0]
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                   create_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                                    create_graph=True)[0]
        
        # Ecuación de Black-Scholes
        residual = (V_t + 0.5 * self.sigma**2 * S**2 * V_SS + 
                   self.r * S * V_S - self.r * V)
        
        return residual
    
    def sample_points(self, n_interior, n_boundary, n_terminal):
        """
        Genera puntos de entrenamiento
        """
        S_min, S_max = self.S_range
        t_min, T = self.t_range
        
        # Puntos interiores (PDE)
        S_int = torch.rand(n_interior, 1) * (S_max - S_min) + S_min
        t_int = torch.rand(n_interior, 1) * (T - t_min) + t_min
        
        # Puntos en condición terminal (t = T)
        S_term = torch.rand(n_terminal, 1) * (S_max - S_min) + S_min
        t_term = torch.ones(n_terminal, 1) * T
        
        # Puntos en frontera S = 0
        S_bound_low = torch.zeros(n_boundary, 1)
        t_bound_low = torch.rand(n_boundary, 1) * (T - t_min) + t_min
        
        # Puntos en frontera S = S_max (aproximación de infinito)
        S_bound_high = torch.ones(n_boundary, 1) * S_max
        t_bound_high = torch.rand(n_boundary, 1) * (T - t_min) + t_min
        
        return (S_int.to(device), t_int.to(device), 
                S_term.to(device), t_term.to(device),
                S_bound_low.to(device), t_bound_low.to(device),
                S_bound_high.to(device), t_bound_high.to(device))
    
    def loss_function(self, S_int, t_int, S_term, t_term, 
                     S_bound_low, t_bound_low, S_bound_high, t_bound_high):
        """
        Función de pérdida total: PDE + condiciones de frontera
        """
        # Loss de la PDE (interior)
        pde_residual = self.black_scholes_pde(S_int, t_int)
        loss_pde = torch.mean(pde_residual**2)
        
        # Loss de condición terminal
        V_term = self.model(S_term, t_term)
        payoff_term = self.payoff(S_term)
        loss_terminal = torch.mean((V_term - payoff_term)**2)
        
        # Loss de frontera inferior (S = 0)
        V_bound_low = self.model(S_bound_low, t_bound_low)
        loss_bound_low = torch.mean(V_bound_low**2)
        
        # Loss de frontera superior (S → ∞)
        V_bound_high = self.model(S_bound_high, t_bound_high)
        # Para call: V ≈ S - K*exp(-r*(T-t))
        if self.option_type == 'call':
            V_expected = S_bound_high - self.K * torch.exp(-self.r * (self.t_range[1] - t_bound_high))
        else:
            V_expected = torch.zeros_like(V_bound_high)
        loss_bound_high = torch.mean((V_bound_high - V_expected)**2)
        
        # Loss total con pesos
        total_loss = (loss_pde + 
                     10.0 * loss_terminal + 
                     1.0 * loss_bound_low + 
                     1.0 * loss_bound_high)
        
        return total_loss, loss_pde, loss_terminal, loss_bound_low, loss_bound_high
    
    def train(self, epochs, n_interior=1000, n_boundary=100, n_terminal=100):
        """
        Loop de entrenamiento
        """
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Muestrear puntos
            points = self.sample_points(n_interior, n_boundary, n_terminal)
            
            # Calcular loss
            total_loss, loss_pde, loss_term, loss_low, loss_high = self.loss_function(*points)
            
            # Backpropagation
            total_loss.backward()
            self.optimizer.step()
            
            self.losses.append(total_loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  PDE Loss: {loss_pde.item():.6f}")
                print(f"  Terminal Loss: {loss_term.item():.6f}")
                print(f"  Boundary Low: {loss_low.item():.6f}")
                print(f"  Boundary High: {loss_high.item():.6f}")
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


def analytical_black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Solución analítica de Black-Scholes para comparación
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


# ==================== EJEMPLO DE USO ====================
if __name__ == "__main__":
    # Parámetros del mercado
    K = 100.0      # Strike price
    r = 0.05       # Risk-free rate (5%)
    sigma = 0.2    # Volatility (20%)
    T = 1.0        # Time to maturity (1 año)
    
    # Rangos
    S_range = [50.0, 150.0]
    t_range = [0.0, T]
    
    # Crear modelo
    model = BlackScholesPINN(layers=[2, 64, 64, 64, 1])
    
    # Crear trainer
    trainer = BlackScholesTrainer(
        model=model,
        S_range=S_range,
        t_range=t_range,
        K=K,
        r=r,
        sigma=sigma,
        option_type='call'
    )
    
    # Entrenar
    print("Entrenando PINN...")
    trainer.train(epochs=2000, n_interior=2000, n_boundary=200, n_terminal=200)
    
    # Visualización 1: Comparación con solución analítica
    S_test = np.linspace(50, 150, 100)
    t_test = np.ones_like(S_test) * 0.5  # Evaluamos en t = 0.5
    
    # Predicciones PINN
    V_pinn = trainer.predict(S_test, t_test).flatten()
    
    # Solución analítica
    V_analytical = analytical_black_scholes(S_test, K, T - 0.5, r, sigma, 'call')
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(S_test, V_pinn, label='PINN', linewidth=2)
    plt.plot(S_test, V_analytical, '--', label='Black-Scholes Analítico', linewidth=2)
    plt.xlabel('Precio del Activo (S)')
    plt.ylabel('Precio de la Opción (V)')
    plt.title('Comparación PINN vs Analítico\n(t = 0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Visualización 2: Error
    plt.subplot(1, 3, 2)
    error = np.abs(V_pinn - V_analytical)
    plt.plot(S_test, error, color='red', linewidth=2)
    plt.xlabel('Precio del Activo (S)')
    plt.ylabel('Error Absoluto')
    plt.title('Error PINN vs Analítico')
    plt.grid(True, alpha=0.3)
    
    # Visualización 3: Curva de aprendizaje
    plt.subplot(1, 3, 3)
    plt.plot(trainer.losses)
    plt.xlabel('Época')
    plt.ylabel('Loss Total')
    plt.title('Curva de Aprendizaje')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('black_scholes_pinn_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Métricas finales
    mse = np.mean((V_pinn - V_analytical)**2)
    mae = np.mean(np.abs(V_pinn - V_analytical))
    print(f"\n{'='*50}")
    print(f"MÉTRICAS FINALES:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Error relativo promedio: {np.mean(error/V_analytical)*100:.2f}%")
    print(f"{'='*50}")