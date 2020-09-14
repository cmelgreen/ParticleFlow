Explanation of my rewrite copied from Github.
The original posters question and code are at:[Modelling Of Particle Flow Under Electrical Field](https://codereview.stackexchange.com/questions/249239/modelling-of-partcle-flow-under-electric-field)



**TLDR: Vectorize and eliminate redundant calculations. Rewriting most of your code gave a 400-500x speed up**

Also, use functions and name your variables in ways that people can read

**Long version:**

Legibility
The code in ```ParticleFlow```, ```ExplicitEuler```, and ```Integrate``` was well organize but slow and hard to follow. The variable names were cryptic. They get repeated in different functions with different meanings. Method parameters and global variables are intermixed randomly. And one case a variable is created, explicitly passed through 4 functions, and then thrown away without ever being using.

If there hadn't been an academic paper for reference I don't think I could have figured out what was happening beyond something is getting integrated.

Then the second half of the code was hard to follow because organizing functions and comments suddenly vanish. For example:
```
for i in range(lv):
    ri = ri + temp[i*nop:(i+1)*nop]*dt
    r = np.append(r, ri, axis =0)
```
Is there without any explanation of what it does or why. What is temp and why is it getting sliced between ```i*nop``` and ```(i+1)*nop```? In the rewrite I tried to give examples of better naming and structure so I'll stop harping on legibility now and move on the the interesting stuff of

Performance
Once I read through the paper the code was based on on I could follow the gist of what it calculates and it found it does so pretty inefficiently. Running on my laptop it takes about 3.4 seconds to use Euler's Method to approximate the solution for 100 particles. The rewrite below does the exact same thing in 0.007 seconds.

The major performance issues are that you repeat calculations based on static variables hundreds/thousands of times and you repeat them in the slowest way possible so your method has to slog through the same number crunching again and again and again.

Look at cd():
```
   def cd():
        re = (self.rf*np.linalg.norm(v0, axis = 1)*self.d)/self.na
        t = np.zeros(nop)
        for i, j in enumerate(re):
            t[i] = (24/j)*(1 + 0.15*pow(j,0.681)) + 0.407/(1+(8710/j))
        return t
```
The beauty of numpy is that it lets you performed vectorized calculations entire arrays at once. That for loop isn't doing anything that couldn't be written in one line. And once you start using vectorized results you can strip out the overhead of creating empty ```N```-length vectors (or rather ```nop```-length, an ambiguously named global variable defined halfway down) and filling it element-by-element every iteration. Vectorize whatever you can. It'll give roughly an order of magnitude speedup for free.
```
def fe():
    fec = np.zeros(v0.shape)
    fec[:,1] = np.pi*pow(self.d,2)*self.ec*self.el
    return fec
```
Is better but you're still creating things inefficiently. ```np.zeros(v0.shape)``` could be ```np.zeros_like(v0)```. You're only using values in ```fec[:,1]``` but are creating a 2D array to hold them. But, the thing that really kills efficiency: You go through the whole process of creating an unnecessarily large array, putting some values in it, and then returning it 4,000 separate times for the same result.

The differential equation you're approximating a solution to only depends on the variable ```V```. Everything in ```fg``` is a constant defined at run time. That whole function can be rewritten as a single line that gets run once on initialization and saved:
```
self.fe = np.pi*pow(self.p.d, 2)*self.p.ec*self.env.el
```
Cutting out out almost a million operations (2D array x 100 elements x 4000 calls = 800,000 allocations) just for the empty array. If you increase the number of particles it saves even more. I can see how you copied the equations from the article but pay attention to what changes and what does to see what you can calculate once and save.

This next example I'm not sure sped anything up much but highlights a way of thinking that can help save time in the long run. You wrote ```ExplicitEuler``` as a class:
```
class ExplicitEuler:
    """
    Euler scheme for the numerical resolution of 
    a differentiel equation.
    """
    def __init__(self,f):
        self.f = f

    def iterate(self,v0,t,dt):
        return v0+dt*self.f(v0,t)
```
When you look at the code that runs it though it becomes apparent that again, nothing it does ever changes. There's no actual state being used. When you start looking for bits and pieces like that you can realize that all you actually want from it is an interface to define a specific function composition (a function using a function). You don't need a class with initialization and methods to do that. You can use a simple function:
```
def ExplicitEuler(func):
    def f(v, dt):
        return v+dt*func(v)
    return f
```
Same thing with ```Integrator```. It gets passed some variables once and returns a single result. You don't need a class for it. You can even change them both to functions still keep your moneyshot line:
```
eul = Integrator(ExplicitEuler(ParticleFlow(d0, ec0, 1e05, rp, rf, g, na)),vint,tmin,tmax,2000)
temp = eul.integrate()
```
Becomes
```
vv = Integrate(ExplicitEuler(ParticleFlow(p, e)), p.v0, tmin, tmax, nop)
with just a bit of refactoring.
```
You can read through my full rewrite below. There are a few odds and ends I won't go into detail on (grouping variables as tuples, adding some functions to spaghetti code sections, adding human readable names and comments, etc).

But again, the three big things to pay attention to are: **Vectorization, Redudant/Expensive Calculation, and Legibility**
