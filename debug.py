from model import CharToPhonModel

m = CharToPhonModel(data_dir="data/",
                batch_size=16,
                embed_dims=100,
                hidden_dims=100,
                bidir=True,
                # cell_class=LSTMCell,
                max_gradient_norm=1,
                learning_rate=0.001,
                save_dir="output_1/",
                resume_dir=None,
                n_batches=10001,
                debug=False,
                print_every=100,
                validate_every=1000
                )

# m = CharToPhonModel(data_dir="data/",
#                 batch_size=4,
#                 embed_dims=100,
#                 hidden_dims=100,
#                 bidir=True,
#                 # cell_class=LSTMCell,
#                 max_gradient_norm=1,
#                 learning_rate=0.001,
#                 save_dir="output_1/",
#                 resume_dir=None,
#                 n_batches=501,
#                 debug=False,
#                 print_every=10,
#                 validate_every=100
#                 )


# m.train()
m.inference()

