from model import CharToPhonModel

m = CharToPhonModel(
                data_dir="data/",
                batch_size=9,
                embed_dims=50,
                hidden_dims=50,
                bidir=True,
                max_gradient_norm=1,
                learning_rate=0.001,
                save_dir="unsaved_model/",
                resume_dir=None,
                n_batches=501,
                debug=True,
                print_every=50,
                validate_every=100,
                beam_search=False,
                beam_width=10
                )


m.train()
m.inference()

