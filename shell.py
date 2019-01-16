import cmd

class CustomShell(cmd.Cmd):
    intro = 'Custom intro for shell'
    prompt = '(Custom) '

    def do_test(self, arg):
        'Test of the custom cmd'
        print(arg)

    def do_train(self, arg):
        'Trains the model for poster generation'
        pass

    def do_generate(self, arg):
        'Generate a poster from the type given'
        pass

    def do_infos(self, arg):
        'Get the infos about the poster'
        self.usage('infos',['input_size','output_size','channels'])

    def do_quit(self, arg):
        'Leave command'
        return True

    def usage(self, command, parameters):
        print('Usage : %s' % command, end=' ')
        for i in range(len(parameters)):
            print('<%s>' % parameters[i], end=' ')
        print('')

CustomShell().cmdloop()
