import numpy as np

class DataGen:
    def trivial_humano(self, n):
        X = []
        y = []
        
        y_axis = np.linspace(-2, 2, n)
        x1 = np.linspace(-1.5, -0.5, n)
        x2 = np.linspace(0.5, 1.5, n)

        x1v, y1v = np.meshgrid(x1, y_axis)
        x2v, y2v = np.meshgrid(x2, y_axis)

        for i in range(n):
            for j in range(n):
                
                _x1 = x1v[j,i]
                _y1 = y1v[j,i]

                _x2 = x2v[j,i]
                _y2 = y2v[j,i]

                X.append( (_x1, _y1) )
                y.append([1, 0])
                
                X.append( (_x2, _y2) )
                y.append([0, 1])

        return np.array(X), np.array(y)

    def uma_ilha(self, n=30):
        ## Gera n pontos equidistantes no intervalo [-3, 3]
        x_range = np.linspace(-3, 3, n)
        ## Cria um grid com [-3, 3] x [-3, 3]
        xx, yy = np.meshgrid(x_range, x_range, indexing="ij")

        x = []
        y = []

        for i in range(len(xx)):
            for j in range(len(yy)):
                dist = np.power(xx[i,j], 2) + np.power(yy[i,j], 2)

                ## Pontos no circulo de raio 2
                if dist <= 2:
                    ## Essa verificação se faz necessária para criar uma margem
                    ## entre a ilha e os pontos ao seu redor. Neste caso, a distência 
                    ## é de 0.7
                    if dist <= 1.3:
                        y.append((0, 1))
                        x.append((xx[i, j], yy[i,j]))
                else:
                    if np.random.uniform(0,1) >= 0.4: continue
                    y.append((1,0))
                    x.append((xx[i, j], yy[i,j]))

        return np.array(x), np.array(y)
