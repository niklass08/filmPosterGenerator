import cmd
from gan import Gan
from filmPosterGan import filmPosterGan

class CustomShell(cmd.Cmd):
    intro = 'Custom intro for shell'
    prompt = '(Custom) '
    fpg = filmPosterGan(rows = 32, cols = 32, channels = 3, dataFolder = "./data")

    def do_generate(self, arg):
        'Generate a poster from the type given'
        args = parse(arg)
        l = len(args)
        if(l != 3):
            usage('generate',['model_name','folder','nb_images'])
            return
        model_name = args[0]
        folder = args[1]
        nb_images = args[2]
        fpg.generate(model_name=model_name,folder=folder,nb_images=nb_images)

    def do_train(self, arg):
        'Trains the model for poster generation'
        args = parse(arg)
        l = len(args)
        if(l != 2):
            usage('train',['N_EPOCHS','model_name'])
            return
        N_EPOCHS = args[0]
        model_name = args[1]
        N_DATA = 3866
        fpg.train(N_EPOCHS=N_EPOCHS, NB_DATA=N_DATA, model_name=model_name)

    def do_quit(self, arg):
        'Leave command'
        return True

def usage(command, parameters):
    print('Usage : %s' % command, end=' ')
    for i in range(len(parameters)):
        print('<%s>' % parameters[i], end=' ')
    print('')

def parse(arg):
    'Convert the arg line to list of args'
    return list(map(str, arg.split()))

CustomShell().cmdloop()
